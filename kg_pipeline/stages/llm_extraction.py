from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from kg_pipeline.models.types import ChunkRecord, KGTriple, NEREntityCandidate, kg_triple_array_schema
from kg_pipeline.prompts.extraction_prompt import build_extraction_prompt
from kg_pipeline.utils.acronym_map import update_acronym_map
from kg_pipeline.utils.validation import parse_json_array, validate_triples, write_failed_chunk


_GENERIC_SECTION_TITLES = {
    "abstract",
    "acknowledgements",
    "acknowledgments",
    "annex",
    "appendix",
    "background",
    "bibliography",
    "conclusion",
    "contents",
    "discussion",
    "executive summary",
    "foreword",
    "introduction",
    "methods",
    "methodology",
    "preface",
    "references",
    "results",
    "summary",
    "table of contents",
}

_SECTION_PREFIX_RE = re.compile(r"^(annex|appendix|chapter|section|part)\s+\w+", re.IGNORECASE)


def _build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url.rstrip("/"), api_key=api_key or "EMPTY")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _log_new_label(new_label_path: Path, label: str) -> None:
    new_label_path.parent.mkdir(parents=True, exist_ok=True)
    with new_label_path.open("a", encoding="utf-8") as f:
        f.write(label.strip() + "\n")


def _normalize_title(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _looks_like_section_title(title: str, section_title: str) -> bool:
    title_norm = _normalize_title(title)
    if not title_norm:
        return False

    section_norm = _normalize_title(section_title)
    if section_norm in {"", "smalldoc", "full document"}:
        section_norm = ""

    if section_norm and title_norm == section_norm:
        return True
    if title_norm in _GENERIC_SECTION_TITLES:
        return True
    if _SECTION_PREFIX_RE.match(title_norm):
        return True
    return False


def _entity_title(entity_text: str, props: dict[str, Any]) -> str:
    value = props.get("title") or props.get("name") or entity_text
    return str(value or "")


def _enforce_labels(
    triple: KGTriple,
    allowed_labels: set[str],
    new_label_log_path: Path,
    section_title: str,
) -> KGTriple:
    def normalize_labels(labels: list[str], entity_title: str) -> list[str]:
        cleaned = [label.strip() for label in labels if label.strip()]
        if not cleaned:
            cleaned = ["Concept"]
        output: list[str] = []
        for label in cleaned:
            if label not in allowed_labels:
                _log_new_label(new_label_log_path, label)
                continue
            output.append(label)

        if "Document" in output and _looks_like_section_title(entity_title, section_title):
            output = [label for label in output if label != "Document"]

        return output or ["Concept"]

    triple.subject_labels = normalize_labels(
        triple.subject_labels,
        _entity_title(triple.subject, triple.subject_properties),
    )
    triple.object_labels = normalize_labels(
        triple.object_labels,
        _entity_title(triple.object, triple.object_properties),
    )
    return triple


def _llm_call(
    client: OpenAI,
    model_name: str,
    prompt: str,
    temperature: float,
    seed: int,
    use_structured_output: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "seed": seed,
        "messages": [{"role": "user", "content": prompt}],
    }

    if use_structured_output:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "kg_triples",
                "schema": kg_triple_array_schema(),
            },
        }

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def extract_triples(
    chunks: list[ChunkRecord],
    ner_map: dict[str, list[NEREntityCandidate]],
    allowed_labels: list[str],
    base_url: str,
    model_name: str,
    api_key: str,
    max_retries_per_chunk: int,
    temperature: float,
    seed: int,
    use_structured_output: bool,
    failed_chunks_path: Path,
    new_label_log_path: Path,
) -> tuple[list[KGTriple], dict[str, str]]:
    client = _build_client(base_url=base_url, api_key=api_key)
    allowed_label_set = set(allowed_labels)

    all_triples: list[KGTriple] = []
    acronym_map: dict[str, str] = {}

    for chunk in tqdm(chunks, desc="Stage 3 LLM Extraction", unit="chunk"):
        candidates = [entity.model_dump() for entity in ner_map.get(chunk.chunk_id, [])]
        prompt = build_extraction_prompt(chunk, candidates, allowed_labels)

        update_acronym_map(acronym_map, chunk.text)
        for entity in candidates:
            update_acronym_map(acronym_map, entity.get("text_span", ""))

        success = False
        for attempt in range(1, max_retries_per_chunk + 1):
            try:
                raw = _llm_call(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    temperature=temperature,
                    seed=seed,
                    use_structured_output=use_structured_output,
                )

                parsed = parse_json_array(raw)
                validated = validate_triples(parsed)
                cleaned_triples: list[KGTriple] = []

                for triple in validated:
                    triple = _enforce_labels(
                        triple,
                        allowed_label_set,
                        new_label_log_path,
                        section_title=chunk.section_title,
                    )
                    rel = dict(triple.relationship_properties)
                    rel.setdefault("source_doc", chunk.filename)
                    rel.setdefault("extraction_method", "llm")
                    rel.setdefault("chunk_id", chunk.chunk_id)
                    rel.setdefault("page_range", chunk.page_range)
                    triple.relationship_properties = rel
                    cleaned_triples.append(triple)

                all_triples.extend(cleaned_triples)
                success = True
                break

            except Exception as exc:
                write_failed_chunk(
                    failed_path=failed_chunks_path,
                    chunk_metadata=chunk.model_dump(),
                    attempt=attempt,
                    error=str(exc),
                    raw_response=locals().get("raw", ""),
                )

        if not success:
            continue

    return all_triples, acronym_map


def save_triples(path: Path, triples: list[KGTriple]) -> None:
    payload = [triple.as_dict() for triple in triples]
    _save_json(path, payload)


def load_triples(path: Path) -> list[KGTriple]:
    payload = _load_json(path)
    return [KGTriple.model_validate(item) for item in payload]


def save_acronyms(path: Path, acronym_map: dict[str, str]) -> None:
    _save_json(path, acronym_map)


def load_acronyms(path: Path) -> dict[str, str]:
    return _load_json(path)


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-json", required=True)
    parser.add_argument("--ner-json", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--output-triples-json", required=True)
    parser.add_argument("--output-acronyms-json", required=True)
    parser.add_argument("--failed-chunks-jsonl", required=True)
    parser.add_argument("--new-label-log", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structured-output", action="store_true")
    args = parser.parse_args()

    chunks_payload = _load_json(Path(args.chunks_json))
    ner_payload = _load_json(Path(args.ner_json))
    labels = _load_json(Path(args.labels_json))

    chunks = [ChunkRecord.model_validate(item) for item in chunks_payload]
    ner_map = {
        chunk_id: [NEREntityCandidate.model_validate(e) for e in entities]
        for chunk_id, entities in ner_payload.items()
    }

    triples, acronym_map = extract_triples(
        chunks=chunks,
        ner_map=ner_map,
        allowed_labels=labels,
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        max_retries_per_chunk=args.max_retries,
        temperature=args.temperature,
        seed=args.seed,
        use_structured_output=args.structured_output,
        failed_chunks_path=Path(args.failed_chunks_jsonl),
        new_label_log_path=Path(args.new_label_log),
    )

    save_triples(Path(args.output_triples_json), triples)
    save_acronyms(Path(args.output_acronyms_json), acronym_map)


if __name__ == "__main__":
    _cli()
