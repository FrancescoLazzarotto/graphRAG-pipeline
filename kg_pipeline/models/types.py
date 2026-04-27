from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SEED_ONTOLOGY_LABELS = [
    "Region",
    "Commodity",
    "Indicator",
    "DataValue",
    "Policy",
    "Organization",
    "Event",
    "Concept",
    "TimePeriod",
    "Document",
    "Dataset",
    "Method",
]

_PREDICATE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


class SectionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    level: int = Field(ge=1, le=6)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)


class PageChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_number: int = Field(ge=1)
    text: str


class DocumentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    filename: str
    page_count: int = Field(ge=1)
    markdown_text: str
    sections: list[SectionRecord] = Field(default_factory=list)
    page_chunks: list[PageChunkRecord] = Field(default_factory=list)
    title: str | None = None
    publication_year: int | None = None


class ChunkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_id: str
    filename: str
    chunk_id: str
    page_range: str
    section_title: str
    chunk_index: int = Field(ge=1)
    text: str


class NEREntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_span: str
    entity_label: str
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    confidence_score: float = Field(ge=0.0, le=1.0)


class KGTriple(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str
    predicate: str
    object: str
    subject_labels: list[str]
    object_labels: list[str]
    subject_properties: dict[str, Any]
    object_properties: dict[str, Any]
    relationship_properties: dict[str, Any]

    @field_validator("subject", "object")
    @classmethod
    def _trim_entity_names(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("subject/object cannot be empty")
        return cleaned

    @field_validator("subject_labels", "object_labels")
    @classmethod
    def _normalize_labels(cls, labels: list[str]) -> list[str]:
        cleaned = [label.strip() for label in labels if str(label).strip()]
        if not cleaned:
            return ["Concept"]
        return cleaned

    @field_validator("predicate")
    @classmethod
    def _validate_predicate(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not _PREDICATE_RE.fullmatch(cleaned):
            raise ValueError("predicate must be SCREAMING_SNAKE_CASE")
        return cleaned

    @model_validator(mode="after")
    def _normalize_properties(self) -> "KGTriple":
        if "name" not in self.subject_properties:
            self.subject_properties["name"] = self.subject
        if "name" not in self.object_properties:
            self.object_properties["name"] = self.object
        if "source_doc" not in self.relationship_properties:
            self.relationship_properties["source_doc"] = ""
        if "extraction_method" not in self.relationship_properties:
            self.relationship_properties["extraction_method"] = "llm"
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "subject_labels": self.subject_labels,
            "object_labels": self.object_labels,
            "subject_properties": self.subject_properties,
            "object_properties": self.object_properties,
            "relationship_properties": self.relationship_properties,
        }


class CanonicalEntityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    canonical_name: str
    aliases: list[str]
    labels: list[str]
    merged_properties: dict[str, Any]
    alias_sources: dict[str, list[str]]


def kg_triple_array_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "subject",
                "predicate",
                "object",
                "subject_labels",
                "object_labels",
                "subject_properties",
                "object_properties",
                "relationship_properties",
            ],
            "properties": {
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "subject_labels": {"type": "array", "items": {"type": "string"}},
                "object_labels": {"type": "array", "items": {"type": "string"}},
                "subject_properties": {"type": "object"},
                "object_properties": {"type": "object"},
                "relationship_properties": {"type": "object"},
            },
        },
    }
