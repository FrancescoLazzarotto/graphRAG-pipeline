from __future__ import annotations

import argparse
import json
from pathlib import Path

from gliner import GLiNER
from tqdm import tqdm

from kg_pipeline.models.types import ChunkRecord, NEREntityCandidate


def run_ner(
    chunks: list[ChunkRecord],
    model_name: str,
    labels: list[str],
    threshold: float,
) -> dict[str, list[NEREntityCandidate]]:
    model = GLiNER.from_pretrained(model_name)

    label_map = {label.lower(): label for label in labels}
    model_labels = list(label_map.keys())

    output: dict[str, list[NEREntityCandidate]] = {}

    for chunk in tqdm(chunks, desc="Stage 2 GLiNER", unit="chunk"):
        raw_entities = model.predict_entities(chunk.text, model_labels, threshold=threshold)
        entities: list[NEREntityCandidate] = []

        for item in raw_entities:
            raw_label = str(item.get("label", "")).strip().lower()
            mapped_label = label_map.get(raw_label, item.get("label", "Concept"))

            start_char = int(item.get("start", item.get("start_char", 0)))
            end_char = int(item.get("end", item.get("end_char", 0)))
            score = float(item.get("score", item.get("confidence", 0.0)))

            entities.append(
                NEREntityCandidate(
                    text_span=str(item.get("text", item.get("span", ""))).strip(),
                    entity_label=str(mapped_label),
                    start_char=max(0, start_char),
                    end_char=max(0, end_char),
                    confidence_score=max(0.0, min(1.0, score)),
                )
            )

        output[chunk.chunk_id] = entities

    return output


def save_ner(path: Path, ner_map: dict[str, list[NEREntityCandidate]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        chunk_id: [entity.model_dump() for entity in entities]
        for chunk_id, entities in ner_map.items()
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_ner(path: Path) -> dict[str, list[NEREntityCandidate]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        chunk_id: [NEREntityCandidate.model_validate(entity) for entity in entities]
        for chunk_id, entities in payload.items()
    }


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--labels-json", required=True)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    chunks_payload = json.loads(Path(args.chunks_json).read_text(encoding="utf-8"))
    chunks = [ChunkRecord.model_validate(item) for item in chunks_payload]
    labels = json.loads(Path(args.labels_json).read_text(encoding="utf-8"))

    ner_map = run_ner(chunks, args.model_name, labels, args.threshold)
    save_ner(Path(args.output_json), ner_map)


if __name__ == "__main__":
    _cli()
