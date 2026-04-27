from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from kg_pipeline.models.types import CanonicalEntityRecord, KGTriple
from kg_pipeline.utils.acronym_map import expand_acronym


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
                "label": (triple.subject_labels[0] if triple.subject_labels else "Concept"),
                "doc": src_doc,
                "properties": dict(triple.subject_properties),
                "predicates": {predicate},
            }
        )
        mentions.append(
            {
                "name": triple.object,
                "label": (triple.object_labels[0] if triple.object_labels else "Concept"),
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
                if _jaccard(set(mentions[idx]["predicates"]), c_predicates) >= context_jaccard_floor:
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
        names = sorted({mentions[i]["name"] for i in idxs}, key=lambda x: (-len(x), x.lower()))
        canonical_names.append(names[0] if names else "")
        canonical_labels.append(mentions[idxs[0]]["label"])

    model = SentenceTransformer(embedding_model)
    emb = model.encode(canonical_names, normalize_embeddings=True)
    sims = np.matmul(emb, emb.T)

    candidates: list[tuple[int, int]] = []
    n = len(canonical_names)
    for i in range(n):
        for j in range(i + 1, n):
            if canonical_labels[i] != canonical_labels[j]:
                continue
            if float(sims[i, j]) > threshold:
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
        left_docs = sorted({mentions[idx]["doc"] for idx in groups[i] if mentions[idx]["doc"]})
        right_docs = sorted({mentions[idx]["doc"] for idx in groups[j] if mentions[idx]["doc"]})
        docs = sorted(set(left_docs + right_docs)) or ["global"]

        left_name = sorted({mentions[idx]["name"] for idx in groups[i]}, key=lambda x: (-len(x), x.lower()))[0]
        right_name = sorted({mentions[idx]["name"] for idx in groups[j]}, key=lambda x: (-len(x), x.lower()))[0]
        label = mentions[groups[i][0]]["label"]

        pair_payload = {
            "left_group": i,
            "right_group": j,
            "left_name": left_name,
            "right_name": right_name,
            "label": label,
            "left_docs": left_docs,
            "right_docs": right_docs,
        }

        for doc in docs:
            by_doc[doc].append(pair_payload)

    for doc, pairs in tqdm(by_doc.items(), desc="Stage 4 LLM Merge Confirm", unit="doc"):
        prompt = f"""
You are resolving cross-document entities for a food-domain knowledge graph.

Document scope: {doc}

For each pair, decide if they refer to the same real-world entity.
Be conservative: if uncertain, return merge=false.
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
            parsed = json.loads(content)
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("merge", False)):
                    left_group = int(item["left_group"])
                    right_group = int(item["right_group"])
                    approved.add(tuple(sorted((left_group, right_group))))
        except Exception:
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
) -> tuple[list[KGTriple], dict[str, CanonicalEntityRecord]]:
    mentions = _build_mentions(triples)
    groups = _initial_groups(mentions, acronym_map, context_jaccard_floor=context_jaccard_floor)

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

    alias_to_canonical: dict[str, str] = {}
    registry: dict[str, CanonicalEntityRecord] = {}

    for group_idxs in merged_group_map.values():
        mention_indices: list[int] = []
        for gidx in group_idxs:
            mention_indices.extend(groups[gidx])

        aliases = sorted({mentions[midx]["name"] for midx in mention_indices}, key=lambda x: (x.lower(), len(x)))
        canonical_name = sorted(aliases, key=lambda x: (-len(x), x.lower()))[0]
        labels = sorted({mentions[midx]["label"] for midx in mention_indices})
        merged_props: dict[str, Any] = {"name": canonical_name}
        alias_sources: dict[str, list[str]] = defaultdict(list)

        for midx in mention_indices:
            alias = mentions[midx]["name"]
            if alias not in alias_to_canonical:
                alias_to_canonical[alias] = canonical_name
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

    resolved_triples: list[KGTriple] = []
    for triple in triples:
        triple.subject = alias_to_canonical.get(triple.subject, triple.subject)
        triple.object = alias_to_canonical.get(triple.object, triple.object)

        if triple.subject in registry:
            triple.subject_properties = dict(registry[triple.subject].merged_properties)
            if not triple.subject_labels:
                triple.subject_labels = list(registry[triple.subject].labels) or ["Concept"]

        if triple.object in registry:
            triple.object_properties = dict(registry[triple.object].merged_properties)
            if not triple.object_labels:
                triple.object_labels = list(registry[triple.object].labels) or ["Concept"]

        resolved_triples.append(triple)

    return resolved_triples, registry


def save_registry(path: Path, registry: dict[str, CanonicalEntityRecord]) -> None:
    payload = {key: value.model_dump() for key, value in registry.items()}
    _save_json(path, payload)


def load_registry(path: Path) -> dict[str, CanonicalEntityRecord]:
    payload = _load_json(path)
    return {key: CanonicalEntityRecord.model_validate(value) for key, value in payload.items()}


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
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--similarity-threshold", type=float, default=0.92)
    parser.add_argument("--context-jaccard-floor", type=float, default=0.10)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="")
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
    )

    save_triples(Path(args.output_triples_json), resolved)
    save_registry(Path(args.output_registry_json), registry)


if __name__ == "__main__":
    _cli()
