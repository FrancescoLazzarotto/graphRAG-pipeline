#!/usr/bin/env python3
"""
KG Evaluator for Neo4j Aura
============================
Esegui con:  python kg_evaluator.py
Richiede:    pip install neo4j rich python-dotenv

Output: kg_report.json  (incollalo su Claude per la valutazione)
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

from neo4j import GraphDatabase
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv


# ── Carica .env del pipeline (kg_pipeline/.env) se presente
ROOT = Path(__file__).resolve().parents[0]
ENV_PATH = ROOT / "kg_pipeline" / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


# ── Configurazione (prese da env dopo load_dotenv)
NEO4J_URI = os.getenv("NEO4J_URI", os.getenv("NEO4J_URL", "neo4j+s://<id>.databases.neo4j.io"))
NEO4J_USER = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "<password>")
SAMPLE_LIMIT = int(os.getenv("KG_EVALUATOR_SAMPLE_LIMIT", "2000"))

console = Console()


def run(session, query, **params):
    return session.run(query, **params).data()


# ── 1. Metriche di base ──────────────────────────────────────────────────────

def basic_counts(session):
    n = run(session, "MATCH (n) RETURN count(n) AS c")[0]["c"]
    r = run(session, "MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    labels = run(session, "CALL db.labels() YIELD label RETURN collect(label) AS l")[0]["l"]
    reltypes = run(session, "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS t")[0]["t"]
    return {
        "node_count": n,
        "edge_count": r,
        "label_count": len(labels),
        "reltype_count": len(reltypes),
        "labels": labels,
        "relationship_types": reltypes,
    }


# ── 2. Distribuzione per label ───────────────────────────────────────────────

def label_distribution(session, labels):
    dist = {}
    for lbl in labels:
        c = run(session, f"MATCH (n:`{lbl}`) RETURN count(n) AS c")[0]["c"]
        dist[lbl] = c
    return dist


# ── 3. Distribuzione per tipo di relazione ───────────────────────────────────

def reltype_distribution(session, reltypes):
    dist = {}
    for rt in reltypes:
        c = run(session, f"MATCH ()-[r:`{rt}`]->() RETURN count(r) AS c")[0]["c"]
        dist[rt] = c
    return dist


# ── 4. Densità del grafo ─────────────────────────────────────────────────────

def graph_density(n, e):
    """Densità diretta: e / (n*(n-1))"""
    if n < 2:
        return 0.0
    return round(e / (n * (n - 1)), 6)


# ── 5. Distribuzione del grado ───────────────────────────────────────────────

def degree_stats(session, limit=SAMPLE_LIMIT):
    # Aura/modern Neo4j disallow size((n)--()) inside functions; use a COUNT aggregation per node instead
    rows = run(session,
        f"MATCH (n) WITH n LIMIT {limit} "
        f"OPTIONAL MATCH (n)-[r]-() "
        f"WITH n, count(r) AS degree "
        f"RETURN degree")
    degrees = [r["degree"] for r in rows]
    if not degrees:
        return {}
    degrees.sort()
    n = len(degrees)
    mean = sum(degrees) / n
    median = degrees[n // 2]
    variance = sum((d - mean) ** 2 for d in degrees) / n
    std = math.sqrt(variance)
    non_zero = [d for d in degrees if d > 0]
    alpha = None
    if non_zero:
        alpha = round(1 + len(non_zero) / sum(math.log(d) for d in non_zero), 3)
    return {
        "sample_size": n,
        "min": degrees[0],
        "max": degrees[-1],
        "mean": round(mean, 2),
        "median": median,
        "std": round(std, 2),
        "power_law_alpha_estimate": alpha,
        "degree_distribution_top10": dict(Counter(degrees).most_common(10)),
    }


# ── 6. Nodi isolati (orfani) ─────────────────────────────────────────────────

def isolated_nodes(session):
    c = run(session, "MATCH (n) WHERE NOT (n)--() RETURN count(n) AS c")[0]["c"]
    return c


# ── 7. Nodi hub (top-k per grado) ────────────────────────────────────────────

def hub_nodes(session, k=10):
    # Use OPTIONAL MATCH + COUNT to compute degree per node (compatible with Aura)
    rows = run(session,
        f"MATCH (n) "
        f"OPTIONAL MATCH (n)-[r]-() "
        f"WITH n, count(r) AS degree "
        f"ORDER BY degree DESC LIMIT {k} "
        f"RETURN labels(n) AS labels, properties(n) AS props, degree")
    return [{"labels": r["labels"], "degree": r["degree"],
             "id_hint": _id_hint(r["props"])} for r in rows]


def _id_hint(props):
    for key in ("name", "title", "id", "uri", "label", "identifier"):
        if key in props:
            v = props[key]
            return str(v)[:80] if v else None
    if props:
        first_key = next(iter(props))
        return f"{first_key}={str(props[first_key])[:60]}"
    return None


# ── 8. Copertura delle proprietà ─────────────────────────────────────────────

def property_coverage(session, labels, limit=SAMPLE_LIMIT):
    coverage = {}
    for lbl in labels:
        rows = run(session,
            f"MATCH (n:`{lbl}`) WITH n LIMIT {limit} "
            "RETURN keys(n) AS ks, count(n) AS c")
        total = sum(r["c"] for r in rows)
        if total == 0:
            coverage[lbl] = {}
            continue
        prop_counts = Counter()
        for r in rows:
            for k in r["ks"]:
                prop_counts[k] += r["c"]
        coverage[lbl] = {
            "total_sampled": total,
            "properties": {k: round(v / total, 3) for k, v in prop_counts.items()},
        }
    return coverage


# ── 9. Componenti connesse ────────────────────────────────────────────────────

def connected_components(session):
    """Approssimazione via WCC (richiede GDS oppure fallback manuale)."""
    try:
        rows = run(session,
            "CALL gds.wcc.stats({nodeQuery: 'MATCH (n) RETURN id(n) AS id', "
            "relationshipQuery: 'MATCH (a)--(b) RETURN id(a) AS source, id(b) AS target'}) "
            "YIELD componentCount, componentDistribution "
            "RETURN componentCount, componentDistribution")
        if rows:
            r = rows[0]
            return {
                "method": "GDS WCC",
                "component_count": r["componentCount"],
                "distribution": r.get("componentDistribution", {}),
            }
    except Exception:
        pass
    try:
        rows = run(session,
            "MATCH (n) "
            "WITH n LIMIT 5000 "
            "CALL apoc.path.subgraphNodes(n, {maxLevel: 99}) YIELD node "
            "RETURN count(DISTINCT node) AS size LIMIT 1")
        if rows:
            return {"method": "APOC (partial)", "largest_component_approx": rows[0]["size"]}
    except Exception:
        pass
    return {"method": "unavailable (no GDS/APOC)", "note": "installa GDS o APOC su Aura per questa metrica"}


# ── 10. Schemi di triple più frequenti ───────────────────────────────────────

def triple_patterns(session, limit=20):
    rows = run(session,
        f"MATCH (a)-[r]->(b) "
        "WITH labels(a)[0] AS src, type(r) AS rel, labels(b)[0] AS tgt "
        f"RETURN src, rel, tgt, count(*) AS freq "
        f"ORDER BY freq DESC LIMIT {limit}")
    return [{"pattern": f"({r['src']})-[{r['rel']}]->({r['tgt']})", "count": r["freq"]}
            for r in rows]


# ── 11. Rapporto nodi:archi (graph sparsity) ──────────────────────────────────

def sparsity_class(n, e):
    if n == 0:
        return "empty"
    ratio = e / n
    if ratio < 1:
        return "very sparse (<1)"
    elif ratio < 3:
        return "sparse (1–3)"
    elif ratio < 10:
        return "moderate (3–10)"
    else:
        return "dense (>10)"


# ── 12. Autoconsistenza dei tipi di relazione ─────────────────────────────────

def reltype_endpoint_consistency(session, reltypes, limit=500):
    consistency = {}
    for rt in reltypes:
        rows = run(session,
            f"MATCH (a)-[r:`{rt}`]->(b) WITH a, b LIMIT {limit} "
            "RETURN labels(a)[0] AS src, labels(b)[0] AS tgt, count(*) AS c")
        patterns = {f"({r['src']})->({r['tgt']})": r["c"] for r in rows}
        consistency[rt] = {
            "distinct_patterns": len(patterns),
            "patterns": patterns,
        }
    return consistency


# ── 13. Metriche di qualità aggregate ───────────────────────────────────────

def quality_scores(report):
    n = report["basic"]["node_count"]
    e = report["basic"]["edge_count"]
    isolated = report["isolated_node_count"]
    deg = report.get("degree_stats", {})

    structural = round(1 - (isolated / n if n else 0), 3)
    connectivity = round(min((e / n) / 5, 1.0), 3) if n else 0
    richness = round(
        min(report["basic"]["label_count"] / 10, 1.0) * 0.5 +
        min(report["basic"]["reltype_count"] / 20, 1.0) * 0.5, 3)

    cons_scores = []
    for rt, data in report.get("reltype_endpoint_consistency", {}).items():
        p = data["distinct_patterns"]
        cons_scores.append(1 / p if p > 0 else 0)
    consistency = round(sum(cons_scores) / len(cons_scores), 3) if cons_scores else None

    return {
        "structural_completeness": structural,
        "connectivity": connectivity,
        "semantic_richness": richness,
        "reltype_consistency": consistency,
        "note": "Punteggi 0–1. Valori più alti = migliore qualità.",
    }


def evaluate(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    report = {}

    steps = [
        ("Conteggi di base",         "basic"),
        ("Distribuzione label",       "label_distribution"),
        ("Distribuzione rel. types",  "reltype_distribution"),
        ("Degree statistics",         "degree_stats"),
        ("Nodi isolati",              "isolated_node_count"),
        ("Hub nodes",                 "hub_nodes"),
        ("Copertura proprietà",       "property_coverage"),
        ("Componenti connesse",       "connected_components"),
        ("Pattern di triple",         "triple_patterns"),
        ("Consistenza rel. types",    "reltype_endpoint_consistency"),
    ]

    with driver.session() as session:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:

            task = progress.add_task("Avvio...", total=len(steps))

            progress.update(task, description=steps[0][0])
            basic = basic_counts(session)
            report["basic"] = basic
            progress.advance(task)

            progress.update(task, description=steps[1][0])
            report["label_distribution"] = label_distribution(session, basic["labels"])
            progress.advance(task)

            progress.update(task, description=steps[2][0])
            report["reltype_distribution"] = reltype_distribution(session, basic["relationship_types"])
            progress.advance(task)

            progress.update(task, description=steps[3][0])
            report["degree_stats"] = degree_stats(session)
            progress.advance(task)

            progress.update(task, description=steps[4][0])
            report["isolated_node_count"] = isolated_nodes(session)
            progress.advance(task)

            progress.update(task, description=steps[5][0])
            report["hub_nodes"] = hub_nodes(session)
            progress.advance(task)

            progress.update(task, description=steps[6][0])
            report["property_coverage"] = property_coverage(session, basic["labels"])
            progress.advance(task)

            progress.update(task, description=steps[7][0])
            report["connected_components"] = connected_components(session)
            progress.advance(task)

            progress.update(task, description=steps[8][0])
            report["triple_patterns"] = triple_patterns(session)
            progress.advance(task)

            progress.update(task, description=steps[9][0])
            report["reltype_endpoint_consistency"] = reltype_endpoint_consistency(
                session, basic["relationship_types"])
            progress.advance(task)

    driver.close()

    n, e = basic["node_count"], basic["edge_count"]
    report["graph_density"] = graph_density(n, e)
    report["sparsity_class"] = sparsity_class(n, e)
    report["edge_node_ratio"] = round(e / n, 3) if n else 0
    report["quality_scores"] = quality_scores(report)

    return report


if __name__ == "__main__":
    console.rule("[bold]KG Evaluator — Neo4j Aura[/bold]")
    console.print(f"URI: [cyan]{NEO4J_URI}[/cyan]")

    report = evaluate(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    out_path = "kg_report_8.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    console.rule("[green]Report salvato[/green]")
    console.print(f"File: [bold]{out_path}[/bold]")
    console.print()
    console.print("[yellow]→ Carica kg_report.json su Claude per la valutazione completa.[/yellow]")

    b = report["basic"]
    qs = report["quality_scores"]
    console.print()
    console.print(f"  Nodi: {b['node_count']:,}  |  Archi: {b['edge_count']:,}  |  "
                  f"Label: {b['label_count']}  |  Tipi rel.: {b['reltype_count']}")
    console.print(f"  Completezza strutturale : {qs['structural_completeness']}")
    console.print(f"  Connettività            : {qs['connectivity']}")
    console.print(f"  Ricchezza semantica     : {qs['semantic_richness']}")
    console.print(f"  Consistenza rel. types  : {qs['reltype_consistency']}")
