from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kg_pipeline.models.types import CanonicalEntityRecord, KGTriple
from datetime import datetime
from kg_pipeline.utils.acronym_map import expand_acronym
from kg_pipeline.utils.validation import parse_json_array


def _parse_llm_json_array(content: str) -> list:
    """Parse a JSON array from LLM output, tolerating fences and prose."""
    try:
        return parse_json_array(content)
    except Exception:
        start = content.find("[")
        end = content.rfind("]")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
        if not isinstance(parsed, list):
            raise ValueError("LLM output is not a JSON array")
        return parsed


LOGGER = logging.getLogger("kg_pipeline")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _norm(text: str) -> str:
    return _NON_ALNUM.sub("", text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    denom = len(a | b)
    if denom == 0:
        return 0.0
    return len(a & b) / float(denom)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_mentions(triples: list[KGTriple]) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []

    for triple in triples:
        src_doc = str(triple.relationship_properties.get("source_doc", "")).strip()
        predicate = triple.predicate

        mentions.append(
            {
                "name": triple.subject,
                "label": (
                    triple.subject_labels[0] if triple.subject_labels else "Concept"
                ),
                "doc": src_doc,
                "properties": dict(triple.subject_properties),
                "predicates": {predicate},
            }
        )
        mentions.append(
            {
                "name": triple.object,
                "label": (
                    triple.object_labels[0] if triple.object_labels else "Concept"
                ),
                "doc": src_doc,
                "properties": dict(triple.object_properties),
                "predicates": {predicate},
            }
        )

    return mentions


def _initial_groups(
    mentions: list[dict[str, Any]],
    acronym_map: dict[str, str],
    context_jaccard_floor: float,
) -> list[list[int]]:
    groups_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)

    for idx, mention in enumerate(mentions):
        expanded = expand_acronym(mention["name"], acronym_map)
        key = (mention["label"], _norm(expanded))
        groups_by_key[key].append(idx)

    groups: list[list[int]] = []
    for _, idxs in groups_by_key.items():
        if len(idxs) == 1:
            groups.append(idxs)
            continue

        local_clusters: list[list[int]] = []
        for idx in idxs:
            placed = False
            for cluster in local_clusters:
                c_predicates = set()
                for cidx in cluster:
                    c_predicates |= set(mentions[cidx]["predicates"])
                if (
                    _jaccard(set(mentions[idx]["predicates"]), c_predicates)
                    >= context_jaccard_floor
                ):
                    cluster.append(idx)
                    placed = True
                    break
            if not placed:
                local_clusters.append([idx])

        groups.extend(local_clusters)

    return groups


def _embedding_candidates(
    mentions: list[dict[str, Any]],
    groups: list[list[int]],
    embedding_model: str,
    threshold: float,
) -> list[tuple[int, int]]:
    if len(groups) < 2:
        return []

    canonical_names: list[str] = []
    canonical_labels: list[str] = []

    for idxs in groups:
        names = sorted(
            {mentions[i]["name"] for i in idxs}, key=lambda x: (-len(x), x.lower())
        )
        canonical_names.append(names[0] if names else "")
        canonical_labels.append(mentions[idxs[0]]["label"])

    model = SentenceTransformer(embedding_model)
    emb = model.encode(canonical_names, normalize_embeddings=True)
    sims = np.matmul(emb, emb.T)

    # Cross-label pairs are kept as candidates (bilingual corpus: the same
    # entity is often typed differently in the two languages, e.g. "food
    # waste"[Product] vs "spreco alimentare"[Process]); the LLM confirmation
    # step receives both labels and decides. They need a stricter similarity
    # floor than same-label pairs, otherwise candidates explode.
    cross_label_threshold = max(threshold, 0.92)
    candidates: list[tuple[int, int]] = []
    n = len(canonical_names)
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(sims[i, j])
            if canonical_labels[i] == canonical_labels[j]:
                if sim > threshold:
                    candidates.append((i, j))
            elif sim > cross_label_threshold:
                candidates.append((i, j))
    return candidates


def _confirm_candidates_with_llm(
    base_url: str,
    api_key: str,
    model_name: str,
    mentions: list[dict[str, Any]],
    groups: list[list[int]],
    candidates: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    if not candidates:
        return set()

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")
    approved: set[tuple[int, int]] = set()

    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for i, j in candidates:
        left_docs = sorted(
            {mentions[idx]["doc"] for idx in groups[i] if mentions[idx]["doc"]}
        )
        right_docs = sorted(
            {mentions[idx]["doc"] for idx in groups[j] if mentions[idx]["doc"]}
        )
        docs = sorted(set(left_docs + right_docs)) or ["global"]

        left_name = sorted(
            {mentions[idx]["name"] for idx in groups[i]},
            key=lambda x: (-len(x), x.lower()),
        )[0]
        right_name = sorted(
            {mentions[idx]["name"] for idx in groups[j]},
            key=lambda x: (-len(x), x.lower()),
        )[0]

        pair_payload = {
            "left_group": i,
            "right_group": j,
            "left_name": left_name,
            "right_name": right_name,
            "left_label": mentions[groups[i][0]]["label"],
            "right_label": mentions[groups[j][0]]["label"],
            "left_docs": left_docs,
            "right_docs": right_docs,
        }

        for doc in docs:
            by_doc[doc].append(pair_payload)

    _CONFIRM_BATCH_SIZE = 40

    doc_batches: list[tuple[str, list[dict[str, Any]]]] = []
    for doc, pairs in by_doc.items():
        for start in range(0, len(pairs), _CONFIRM_BATCH_SIZE):
            doc_batches.append((doc, pairs[start : start + _CONFIRM_BATCH_SIZE]))

    for doc, pairs in tqdm(
        doc_batches, desc="Stage 4 LLM Merge Confirm", unit="batch"
    ):
        prompt = f"""
You are resolving cross-document entities for a knowledge graph about circular economy
and food systems. The corpus is bilingual: the same real-world entity may appear with an
ENGLISH name in one document and an ITALIAN name in another (e.g. "circular economy" and
"economia circolare", "food waste" and "spreco alimentare"). Such cross-language pairs
SHOULD be merged when they denote the same entity or concept.

Document scope: {doc}

For each pair, decide if they refer to the same real-world entity.
Be conservative: if uncertain, return merge=false. Translation equivalence alone is
sufficient only when the meaning is clearly the same.
When left_label and right_label differ, merge only if the two names clearly denote the
same thing despite the typing difference; a concept and a publication named after it
(e.g. "sustainability" vs the journal "Sustainability") are NOT the same entity.
Return only JSON array with objects:
{{"left_group": int, "right_group": int, "merge": true or false}}

Pairs:
{json.dumps(pairs, ensure_ascii=False, indent=2)}
""".strip()

        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.choices[0].message.content or "[]"
            parsed = _parse_llm_json_array(content)
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("merge", False)):
                    left_group = int(item["left_group"])
                    right_group = int(item["right_group"])
                    approved.add(tuple(sorted((left_group, right_group))))
        except Exception as exc:
            LOGGER.warning(
                "LLM merge confirmation failed for doc=%s (%d pairs skipped): %s",
                doc,
                len(pairs),
                exc,
            )
            continue

    return approved


def resolve_entities(
    triples: list[KGTriple],
    acronym_map: dict[str, str],
    embedding_model: str,
    similarity_threshold: float,
    context_jaccard_floor: float,
    base_url: str | None,
    api_key: str | None,
    model_name: str | None,
    crosslabel_log_path: Path | None = None,
) -> tuple[list[KGTriple], dict[str, CanonicalEntityRecord]]:
    mentions = _build_mentions(triples)
    groups = _initial_groups(
        mentions, acronym_map, context_jaccard_floor=context_jaccard_floor
    )

    candidates = _embedding_candidates(
        mentions=mentions,
        groups=groups,
        embedding_model=embedding_model,
        threshold=similarity_threshold,
    )

    approved: set[tuple[int, int]] = set()
    if base_url and model_name:
        approved = _confirm_candidates_with_llm(
            base_url=base_url,
            api_key=(api_key or "EMPTY"),
            model_name=model_name,
            mentions=mentions,
            groups=groups,
            candidates=candidates,
        )

    uf = UnionFind(len(groups))
    for i, j in approved:
        uf.union(i, j)

    merged_group_map: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(groups)):
        merged_group_map[uf.find(idx)].append(idx)

    # Build initial registry (do not finalize alias -> canonical mapping yet)
    registry: dict[str, CanonicalEntityRecord] = {}

    for group_idxs in merged_group_map.values():
        mention_indices: list[int] = []
        for gidx in group_idxs:
            mention_indices.extend(groups[gidx])

        aliases = sorted(
            {mentions[midx]["name"] for midx in mention_indices},
            key=lambda x: (x.lower(), len(x)),
        )
        canonical_name = sorted(aliases, key=lambda x: (-len(x), x.lower()))[0]
        labels = sorted({mentions[midx]["label"] for midx in mention_indices})
        merged_props: dict[str, Any] = {"name": canonical_name}
        alias_sources: dict[str, list[str]] = defaultdict(list)

        for midx in mention_indices:
            alias = mentions[midx]["name"]
            source_doc = mentions[midx]["doc"]
            if source_doc and source_doc not in alias_sources[alias]:
                alias_sources[alias].append(source_doc)
            for key, value in mentions[midx]["properties"].items():
                merged_props.setdefault(key, value)

        registry[canonical_name] = CanonicalEntityRecord(
            canonical_name=canonical_name,
            aliases=aliases,
            labels=labels,
            merged_properties=merged_props,
            alias_sources=dict(alias_sources),
        )

    def _cross_label_merge_registry(
        registry: dict[str, CanonicalEntityRecord],
        log_path: Path | None = None,
    ) -> tuple[dict[str, CanonicalEntityRecord], dict[str, str]]:
        """Merge registry entries that have same normalized name but different labels.

        Returns (updated_registry, alias_to_canonical_map)
        """
        # Most-specific-first for the circular-food ontology; Concept is the
        # fallback and must stay last.
        precedence = [
            "Person",
            "Organization",
            "Place",
            "Event",
            "Project",
            "Policy",
            "Document",
            "Indicator",
            "Method",
            "Product",
            "Material",
            "Process",
            "DataValue",
            "Concept",
        ]
        norm_map: dict[str, list[str]] = defaultdict(list)
        for cname in list(registry.keys()):
            norm = cname.strip().lower()
            if not norm:
                LOGGER.warning(
                    "Registry entry with empty normalized name skipped: %r", cname
                )
                continue
            norm_map[norm].append(cname)

        log_lines: list[str] = []
        for norm, cnames in norm_map.items():
            if len(cnames) < 2:
                continue
            # gather labels
            label_sets = {lbl for cname in cnames for lbl in registry[cname].labels}

            # choose canonical label by precedence; case-variant duplicates with a
            # single shared label are merged as well (same normalized name must
            # map to one canonical entry)
            chosen_label = None
            for p in precedence:
                if p in label_sets:
                    chosen_label = p
                    break
            if not chosen_label:
                chosen_label = sorted(label_sets)[0] if label_sets else "Concept"

            # choose keeper record (prefer record that already contains chosen_label)
            keeper: str | None = None
            candidates = [c for c in cnames if chosen_label in registry[c].labels]
            if candidates:
                keeper = sorted(candidates, key=lambda x: (-len(x), x.lower()))[0]
            else:
                keeper = sorted(cnames, key=lambda x: (-len(x), x.lower()))[0]

            removed = []
            for other in cnames:
                if other == keeper:
                    continue
                # merge aliases and alias_sources
                for a in registry[other].aliases:
                    if a not in registry[keeper].aliases:
                        registry[keeper].aliases.append(a)
                for a, srcs in registry[other].alias_sources.items():
                    registry[keeper].alias_sources.setdefault(a, [])
                    for s in srcs:
                        if s not in registry[keeper].alias_sources[a]:
                            registry[keeper].alias_sources[a].append(s)
                # merge properties (keep existing keys on keeper)
                for k, v in registry[other].merged_properties.items():
                    registry[keeper].merged_properties.setdefault(k, v)
                removed.append({"canonical": other, "labels": registry[other].labels})
                # delete other
                try:
                    del registry[other]
                except KeyError:
                    pass

            # set canonical label on keeper to chosen_label
            registry[keeper].labels = [chosen_label]

            # log
            ts = datetime.utcnow().isoformat()
            entry = {
                "timestamp": ts,
                "normalized_name": norm,
                "keeper": keeper,
                "removed": removed,
                "chosen_label": chosen_label,
            }
            log_lines.append(json.dumps(entry, ensure_ascii=False))

        # write log if requested
        if log_path:
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as fh:
                    for l in log_lines:
                        fh.write(l + "\n")
            except OSError as exc:
                LOGGER.warning(
                    "Could not write cross-label merge log %s: %s", log_path, exc
                )

        # build alias_to_canonical map from final registry
        alias_to_canonical: dict[str, str] = {}
        for cname, rec in registry.items():
            for a in rec.aliases:
                alias_to_canonical[a] = cname

        return registry, alias_to_canonical

    # perform cross-label merging and build alias mapping
    registry, alias_to_canonical = _cross_label_merge_registry(
        registry, log_path=Path(crosslabel_log_path) if crosslabel_log_path else None
    )

    resolved_triples: list[KGTriple] = []
    for triple in triples:
        triple.subject = alias_to_canonical.get(triple.subject, triple.subject)
        triple.object = alias_to_canonical.get(triple.object, triple.object)

        if triple.subject in registry:
            triple.subject_properties = dict(registry[triple.subject].merged_properties)
            if not triple.subject_labels:
                triple.subject_labels = list(registry[triple.subject].labels) or [
                    "Concept"
                ]
            else:
                # prefer registry labels if they exist
                triple.subject_labels = list(registry[triple.subject].labels)

        if triple.object in registry:
            triple.object_properties = dict(registry[triple.object].merged_properties)
            if not triple.object_labels:
                triple.object_labels = list(registry[triple.object].labels) or [
                    "Concept"
                ]
            else:
                triple.object_labels = list(registry[triple.object].labels)

        resolved_triples.append(triple)

    return resolved_triples, registry


def save_registry(path: Path, registry: dict[str, CanonicalEntityRecord]) -> None:
    seen_norms: dict[str, str] = {}
    for key in registry:
        norm = key.strip().lower()
        if norm in seen_norms:
            LOGGER.warning(
                "Registry contains case-variant duplicate canonical names: %r and %r",
                seen_norms[norm],
                key,
            )
        else:
            seen_norms[norm] = key
    payload = {key: value.model_dump() for key, value in registry.items()}
    _save_json(path, payload)


def load_registry(path: Path) -> dict[str, CanonicalEntityRecord]:
    payload = _load_json(path)
    return {
        key: CanonicalEntityRecord.model_validate(value)
        for key, value in payload.items()
    }


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    payload = [triple.as_dict() for triple in triples]
    _save_json(path, payload)


def load_triples(path: Path) -> list[KGTriple]:
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples-json", required=True)
    parser.add_argument("--acronyms-json", required=True)
    parser.add_argument("--output-triples-json", required=True)
    parser.add_argument("--output-registry-json", required=True)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.88)
    parser.add_argument("--context-jaccard-floor", type=float, default=0.15)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--crosslabel-log", default="")
    args = parser.parse_args()

    triples = load_triples(Path(args.triples_json))
    acronym_map = _load_json(Path(args.acronyms_json))

    resolved, registry = resolve_entities(
        triples=triples,
        acronym_map=acronym_map,
        embedding_model=args.embedding_model,
        similarity_threshold=args.similarity_threshold,
        context_jaccard_floor=args.context_jaccard_floor,
        base_url=args.base_url or None,
        api_key=args.api_key or None,
        model_name=args.model_name or None,
        crosslabel_log_path=Path(args.crosslabel_log)
        if args.crosslabel_log
        else Path(args.output_registry_json).resolve().parent
        / "resolution_crosslabel.log",
    )

    save_triples(Path(args.output_triples_json), resolved)
    save_registry(Path(args.output_registry_json), registry)


if __name__ == "__main__":
    _cli()
