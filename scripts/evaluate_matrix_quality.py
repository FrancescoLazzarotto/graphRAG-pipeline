from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _normalize(text: str) -> str:
    normalized = text.lower().replace("’", "'")
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _alias_set(*values: str) -> set[str]:
    return {_normalize(value) for value in values if _normalize(value)}


ALIASES: dict[str, set[str]] = {
    "matrix": _alias_set("The Matrix", "matrix"),
    "keanu": _alias_set("Keanu Reeves", "keanu"),
    "carrie": _alias_set("Carrie-Anne Moss", "carrie anne moss", "carrie moss"),
    "laurence": _alias_set("Laurence Fishburne", "laurence fishburne", "fishburne"),
    "hugo": _alias_set("Hugo Weaving", "hugo weaving", "weaving"),
    "lana": _alias_set("Lana Wachowski", "larry wachowski", "lana"),
    "lilly": _alias_set("Lilly Wachowski", "andy wachowski", "lilly"),
}

ROLE_ALIASES: dict[str, set[str]] = {
    "neo": _alias_set("Neo"),
    "trinity": _alias_set("Trinity"),
    "morpheus": _alias_set("Morpheus"),
    "agent_smith": _alias_set("Agent Smith", "Smith"),
    "architect": _alias_set("Architect"),
}

DIRECTORS = [ALIASES["lana"], ALIASES["lilly"]]
ACTORS = [ALIASES["keanu"], ALIASES["carrie"], ALIASES["laurence"], ALIASES["hugo"]]
ALL_CONNECTED_PEOPLE = [ALIASES["keanu"], ALIASES["carrie"], ALIASES["laurence"], ALIASES["hugo"], ALIASES["lana"], ALIASES["lilly"]]

REL_ACTED = _alias_set("ACTED_IN", "ha recitato", "recitato", "cast", "attore", "attori")
REL_DIRECTED = _alias_set("DIRECTED", "diretto", "regista", "registi")

REFUSAL_MARKERS = [
    _normalize("non è inclusa nei file di domande"),
    _normalize("non e inclusa nei file di domande"),
    _normalize("not directly related to the provided context"),
    _normalize("this information is not directly related to the provided context"),
    _normalize("non è possibile determinare"),
    _normalize("non e possibile determinare"),
    _normalize("non fornisce"),
    _normalize("non specifica esplicitamente"),
]


def _spec(
    kind: str,
    entity_groups: list[set[str]] | None = None,
    literal_groups: list[set[str]] | None = None,
    require_all_entities: bool = True,
    require_all_literals: bool = True,
) -> dict[str, Any]:
    return {
        "evaluable": True,
        "kind": kind,
        "entity_groups": entity_groups or [],
        "literal_groups": literal_groups or [],
        "require_all_entities": require_all_entities,
        "require_all_literals": require_all_literals,
    }


def _unevaluable(reason: str) -> dict[str, Any]:
    return {
        "evaluable": False,
        "kind": "unevaluable",
        "reason": reason,
        "entity_groups": [],
        "literal_groups": [],
        "require_all_entities": True,
        "require_all_literals": True,
    }


def _question_spec(question: str) -> dict[str, Any]:
    qn = _normalize(question)

    if "chi ha diretto the matrix" in qn or "chi sono i registi di the matrix" in qn:
        return _spec("directors", entity_groups=DIRECTORS)

    if "quali persone hanno relazione directed con the matrix" in qn:
        return _spec("directed_relation", entity_groups=DIRECTORS, literal_groups=[REL_DIRECTED], require_all_literals=False)

    if "chi ha recitato in the matrix" in qn or "elenca gli attori principali di the matrix" in qn:
        return _spec("actors", entity_groups=ACTORS)

    if "quali persone hanno relazione acted_in con the matrix" in qn:
        return _spec("acted_relation", entity_groups=ACTORS, literal_groups=[REL_ACTED], require_all_literals=False)

    if "chi interpreta neo" in qn:
        return _spec("character_to_actor", entity_groups=[ALIASES["keanu"]])
    if "chi interpreta trinity" in qn:
        return _spec("character_to_actor", entity_groups=[ALIASES["carrie"]])
    if "chi interpreta morpheus" in qn:
        return _spec("character_to_actor", entity_groups=[ALIASES["laurence"]])
    if "chi interpreta agent smith" in qn:
        return _spec("character_to_actor", entity_groups=[ALIASES["hugo"]])

    if "qual e il ruolo di keanu reeves in the matrix" in qn:
        return _spec("actor_to_role", literal_groups=[ROLE_ALIASES["neo"]])
    if "che personaggio interpreta carrie anne moss in the matrix" in qn:
        return _spec("actor_to_role", literal_groups=[ROLE_ALIASES["trinity"]])
    if "che personaggio interpreta laurence fishburne in the matrix" in qn:
        return _spec("actor_to_role", literal_groups=[ROLE_ALIASES["morpheus"]])
    if "che personaggio interpreta hugo weaving in the matrix" in qn:
        return _spec("actor_to_role", literal_groups=[ROLE_ALIASES["agent_smith"]])

    if "qual e la tagline di the matrix" in qn:
        return _spec("tagline", literal_groups=[_alias_set("welcome to the real world")])

    if "in che anno e uscito the matrix" in qn:
        return _spec("release_year", literal_groups=[_alias_set("1999")])

    if "quali persone nate nel 1967" in qn:
        return _spec("born_1967", entity_groups=[ALIASES["carrie"], ALIASES["lilly"]])
    if "quali persone nate nel 1965" in qn:
        return _spec("born_1965", entity_groups=[ALIASES["lana"]])
    if "quali persone nate nel 1964" in qn:
        return _spec("born_1964", entity_groups=[ALIASES["keanu"]])
    if "quali persone nate nel 1961" in qn:
        return _spec("born_1961", entity_groups=[ALIASES["laurence"]])

    if "quali relazioni partono da keanu reeves verso the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["keanu"], ALIASES["matrix"]], literal_groups=[REL_ACTED], require_all_literals=False)
    if "quali relazioni partono da carrie anne moss verso the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["carrie"], ALIASES["matrix"]], literal_groups=[REL_ACTED], require_all_literals=False)
    if "quali relazioni partono da laurence fishburne verso the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["laurence"], ALIASES["matrix"]], literal_groups=[REL_ACTED], require_all_literals=False)
    if "quali relazioni partono da hugo weaving verso the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["hugo"], ALIASES["matrix"]], literal_groups=[REL_ACTED], require_all_literals=False)

    if "quali relazioni collegano lana wachowski a the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["lana"], ALIASES["matrix"]], literal_groups=[REL_DIRECTED], require_all_literals=False)
    if "quali relazioni collegano lilly wachowski a the matrix" in qn:
        return _spec("relation_person_movie", entity_groups=[ALIASES["lilly"], ALIASES["matrix"]], literal_groups=[REL_DIRECTED], require_all_literals=False)

    if "chi ha sia diretto sia lavorato sul film the matrix" in qn:
        return _spec("directors", entity_groups=DIRECTORS)

    if "chi sono i vicini di the matrix nel grafo" in qn:
        return _spec("neighbors_matrix", entity_groups=ALL_CONNECTED_PEOPLE)

    if "chi sono i nodi persona collegati direttamente a the matrix" in qn:
        return _spec("direct_person_neighbors", entity_groups=ALL_CONNECTED_PEOPLE)

    if "quali attori del film the matrix hanno un ruolo esplicito nel grafo" in qn:
        return _spec("actors_with_roles", entity_groups=ACTORS)

    if "quali soggetti puntano a the matrix nei tripletti" in qn or "quali coppie persona film sono presenti nel grafo" in qn:
        return _spec("person_movie_pairs", entity_groups=ALL_CONNECTED_PEOPLE)

    if "quali relazioni mostrano che the matrix e un film diretto da due persone" in qn:
        return _spec(
            "directed_evidence",
            entity_groups=DIRECTORS,
            literal_groups=[REL_DIRECTED],
            require_all_literals=False,
        )

    if "quali evidenze nel grafo supportano che keanu reeves e nel cast di the matrix" in qn:
        return _spec(
            "cast_evidence",
            entity_groups=[ALIASES["keanu"], ALIASES["matrix"]],
            literal_groups=[REL_ACTED],
            require_all_literals=False,
        )

    if "quali evidenze nel grafo supportano che lana wachowski e lilly wachowski hanno diretto the matrix" in qn:
        return _spec(
            "directed_evidence",
            entity_groups=[ALIASES["lana"], ALIASES["lilly"], ALIASES["matrix"]],
            literal_groups=[REL_DIRECTED],
            require_all_literals=False,
        )

    return _unevaluable("question pattern not covered by factual evaluator")


def _load_metadata(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        parsed = json.loads(raw_value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _iter_result_files(root: Path, tag_contains: str) -> list[Path]:
    if root.is_file() and root.name == "results.csv":
        return [root]

    if root.is_dir() and (root / "results.csv").exists():
        return [root / "results.csv"]

    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")

    files: list[Path] = []
    for candidate in sorted(root.glob("*/results.csv")):
        run_dir = candidate.parent.name
        if tag_contains and tag_contains not in run_dir:
            continue
        files.append(candidate)

    if not files:
        raise FileNotFoundError(f"No results.csv files found under: {root}")

    return files


def _match_groups(answer_norm: str, groups: list[set[str]]) -> tuple[list[bool], float]:
    if not groups:
        return [], 1.0

    matches = [any(alias in answer_norm for alias in group) for group in groups]
    coverage = sum(1 for value in matches if value) / len(groups)
    return matches, coverage


def _detect_refusal(answer_norm: str) -> bool:
    return any(marker in answer_norm for marker in REFUSAL_MARKERS)


def _find_phrase_positions(tokens: list[str], phrase_tokens: list[str]) -> list[int]:
    if not phrase_tokens:
        return []

    positions: list[int] = []
    window = len(phrase_tokens)
    for index in range(0, len(tokens) - window + 1):
        if tokens[index : index + window] == phrase_tokens:
            positions.append(index)
    return positions


def _detect_role_contradiction(answer_norm: str) -> bool:
    tokens = answer_norm.split()
    if not tokens:
        return False

    actor_role_expectation = {
        "keanu": "neo",
        "carrie": "trinity",
        "laurence": "morpheus",
        "hugo": "agent_smith",
    }

    proximity_threshold = 4

    for actor_key, expected_role_key in actor_role_expectation.items():
        actor_positions: list[int] = []
        for alias in ALIASES[actor_key]:
            actor_positions.extend(_find_phrase_positions(tokens, alias.split()))
        if not actor_positions:
            continue

        for role_key, role_aliases in ROLE_ALIASES.items():
            if role_key == expected_role_key:
                continue

            role_positions: list[int] = []
            for alias in role_aliases:
                role_positions.extend(_find_phrase_positions(tokens, alias.split()))
            if not role_positions:
                continue

            for actor_index in actor_positions:
                for role_index in role_positions:
                    if abs(actor_index - role_index) <= proximity_threshold:
                        return True

    return False


def _evaluate_answer(question: str, answer: str) -> dict[str, Any]:
    spec = _question_spec(question)
    answer_norm = _normalize(answer)

    refusal = _detect_refusal(answer_norm)
    contradiction = _detect_role_contradiction(answer_norm)

    entity_matches, entity_coverage = _match_groups(answer_norm, spec["entity_groups"])
    literal_matches, literal_coverage = _match_groups(answer_norm, spec["literal_groups"])

    if spec["entity_groups"]:
        entity_ok = all(entity_matches) if spec["require_all_entities"] else any(entity_matches)
    else:
        entity_ok = True

    if spec["literal_groups"]:
        literal_ok = all(literal_matches) if spec["require_all_literals"] else any(literal_matches)
    else:
        literal_ok = True

    strict_correct = bool(spec["evaluable"] and entity_ok and literal_ok and not refusal and not contradiction)

    any_signal = any(entity_matches) or any(literal_matches)
    partial_correct = bool(spec["evaluable"] and (strict_correct or (any_signal and not refusal and not contradiction)))

    return {
        "evaluable": bool(spec["evaluable"]),
        "kind": spec["kind"],
        "reason": spec.get("reason", ""),
        "strict_correct": strict_correct,
        "partial_correct": partial_correct,
        "entity_coverage": entity_coverage,
        "literal_coverage": literal_coverage,
        "contradiction": contradiction,
        "refusal": refusal,
    }


def _load_rows(csv_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            for row in reader:
                metadata = _load_metadata(row.get("metadata_json", ""))
                rows.append(
                    {
                        "run_dir": csv_path.parent.name,
                        "model_id": str(metadata.get("model_id", "unknown")),
                        "framework": str(metadata.get("framework", "unknown")),
                        "run_index": int(metadata.get("run_index", 0) or 0),
                        "strategy": row.get("strategy", ""),
                        "question": row.get("question", ""),
                        "answer": row.get("answer", ""),
                        "latency_ms": float(row.get("latency_ms", "0") or 0.0),
                    }
                )

    return rows


def _group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["model_id"]), str(row["framework"]), str(row["strategy"]))


def _aggregate(evaluated_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        grouped[_group_key(row)].append(row)

    output: list[dict[str, Any]] = []
    for (model_id, framework, strategy), items in sorted(grouped.items(), key=lambda key_values: key_values[0]):
        total_rows = len(items)
        evaluable = [item for item in items if item["evaluable"]]
        evaluable_count = len(evaluable)

        strict_correct = sum(1 for item in evaluable if item["strict_correct"])
        partial_correct = sum(1 for item in evaluable if item["partial_correct"])
        contradictions = sum(1 for item in evaluable if item["contradiction"])
        refusals = sum(1 for item in evaluable if item["refusal"])

        entity_cov_values = [float(item["entity_coverage"]) for item in evaluable]
        literal_cov_values = [float(item["literal_coverage"]) for item in evaluable]
        latency_values = [float(item["latency_ms"]) for item in evaluable]

        output.append(
            {
                "model_id": model_id,
                "framework": framework,
                "strategy": strategy,
                "rows_total": total_rows,
                "rows_evaluable": evaluable_count,
                "evaluable_coverage": (evaluable_count / total_rows) if total_rows else 0.0,
                "strict_correct_count": strict_correct,
                "partial_correct_count": partial_correct,
                "strict_accuracy": (strict_correct / evaluable_count) if evaluable_count else 0.0,
                "partial_accuracy": (partial_correct / evaluable_count) if evaluable_count else 0.0,
                "contradiction_rate": (contradictions / evaluable_count) if evaluable_count else 0.0,
                "refusal_rate": (refusals / evaluable_count) if evaluable_count else 0.0,
                "avg_entity_coverage": mean(entity_cov_values) if entity_cov_values else 0.0,
                "avg_literal_coverage": mean(literal_cov_values) if literal_cov_values else 0.0,
                "avg_latency_ms_evaluable": mean(latency_values) if latency_values else 0.0,
            }
        )

    return output


def _print_table(aggregated: list[dict[str, Any]]) -> None:
    if not aggregated:
        print("No rows found.")
        return

    header = (
        f"{'model_id':<30} {'framework':<13} {'strategy':<18} "
        f"{'rows':>6} {'eval':>6} {'cover':>8} {'strict':>8} {'partial':>8} {'contrad':>8} {'refusal':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in sorted(aggregated, key=lambda item: (item["model_id"], item["framework"], item["strategy"])):
        print(
            f"{str(row['model_id'])[:30]:<30} "
            f"{str(row['framework'])[:13]:<13} "
            f"{str(row['strategy'])[:18]:<18} "
            f"{int(row['rows_total']):>6d} "
            f"{int(row['rows_evaluable']):>6d} "
            f"{float(row['evaluable_coverage']) * 100:>7.2f}% "
            f"{float(row['strict_accuracy']) * 100:>7.2f}% "
            f"{float(row['partial_accuracy']) * 100:>7.2f}% "
            f"{float(row['contradiction_rate']) * 100:>7.2f}% "
            f"{float(row['refusal_rate']) * 100:>7.2f}%"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate factual quality of Matrix experiment outputs")
    parser.add_argument(
        "input",
        nargs="?",
        default="artifacts/experiments",
        help="Path to experiments root, a run folder, or a results.csv file",
    )
    parser.add_argument("--tag-contains", default="", help="Optional run folder substring filter")
    parser.add_argument("--save-summary-json", default="", help="Path to save aggregated JSON")
    parser.add_argument("--save-summary-csv", default="", help="Path to save aggregated CSV")
    parser.add_argument("--save-row-csv", default="", help="Path to save row-level evaluation CSV")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    try:
        csv_files = _iter_result_files(Path(args.input), tag_contains=args.tag_contains.strip())
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    rows = _load_rows(csv_files)
    evaluated_rows: list[dict[str, Any]] = []
    for row in rows:
        evaluation = _evaluate_answer(question=str(row["question"]), answer=str(row["answer"]))
        evaluated_rows.append({**row, **evaluation})

    aggregated = _aggregate(evaluated_rows)

    print(f"Loaded {len(rows)} rows from {len(csv_files)} result file(s).")
    _print_table(aggregated)

    if args.save_summary_json:
        output_path = Path(args.save_summary_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON summary: {output_path}")

    if args.save_summary_csv:
        output_path = Path(args.save_summary_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "model_id",
                    "framework",
                    "strategy",
                    "rows_total",
                    "rows_evaluable",
                    "evaluable_coverage",
                    "strict_correct_count",
                    "partial_correct_count",
                    "strict_accuracy",
                    "partial_accuracy",
                    "contradiction_rate",
                    "refusal_rate",
                    "avg_entity_coverage",
                    "avg_literal_coverage",
                    "avg_latency_ms_evaluable",
                ],
            )
            writer.writeheader()
            for row in aggregated:
                writer.writerow(row)
        print(f"Saved CSV summary: {output_path}")

    if args.save_row_csv:
        output_path = Path(args.save_row_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(
                file_obj,
                fieldnames=[
                    "run_dir",
                    "model_id",
                    "framework",
                    "run_index",
                    "strategy",
                    "question",
                    "latency_ms",
                    "evaluable",
                    "kind",
                    "reason",
                    "strict_correct",
                    "partial_correct",
                    "entity_coverage",
                    "literal_coverage",
                    "contradiction",
                    "refusal",
                    "answer",
                ],
            )
            writer.writeheader()
            for row in evaluated_rows:
                writer.writerow({key: row.get(key, "") for key in writer.fieldnames})
        print(f"Saved row-level CSV: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
