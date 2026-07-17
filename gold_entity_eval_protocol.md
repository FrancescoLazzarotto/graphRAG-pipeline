# Gold Standard — Entity Evaluation Protocol (addendum)

> To be appended to `gold_full_completed.md`. Defines HOW `expected_entities` are
> scored across pipelines that expose entities in different forms. Fixed BEFORE
> running any pipeline, so scoring rules cannot be tuned to the outcome.

---

## 1. Why this section exists

The three pipelines expose retrieved entities in different native forms:

| pipeline | native entity form | already in gold's URI space? |
|----------|-------------------|------------------------------|
| ontology-grounded (external structure) | canonical vocabulary URIs (AGROVOC/ChEBI/CEON) | **yes** — direct URI↔URI comparison |
| graph-RAG (internal structure, Neo4j) | internal graph node IDs + labels | **no** — needs label→URI alignment |
| plain-text RAG (baseline) | none native; entities extracted post-hoc from answer text | **no** — needs extraction + alignment |

Comparing raw internal node IDs against the gold URIs would make any
non-URI-native pipeline fail by construction — not because it missed the concept,
but because its identifier has a different name. To keep the comparison fair, all
pipelines are mapped to a common evaluation space through **one shared, declared
resolver applied symmetrically**.

This alignment step is itself a finding (external structure is natively
interoperable; internal structure requires a downstream mapping to reach the same
interoperability), but that observation belongs in the DISCUSSION, not in the
scoring: the numbers measure *did it retrieve the right concept*, the discussion
observes *at what interoperability cost*.

---

## 2. Two metrics, reported separately

Every entity comparison is reported at two levels. They are NOT interchangeable
and mixing them into one number hides what differs between pipelines.

### 2a. Concept-level (fair retrieval comparison)
- Compares **normalised labels**, not URIs.
- Question answered: *did the pipeline retrieve the right concept at all?*
- Uses ALL `expected_entities` (both `exact` and `benchmark_local_extension`).
- This is the honest apples-to-apples retrieval metric across pipelines.

### 2b. Grounding-level (interoperability / auditability)
- Compares **resolved canonical URIs**.
- Question answered: *is the retrieved entity anchored to a canonical,
  externally-resolvable identifier?*
- Uses ONLY `expected_entities` with `mapping_status: exact` (i.e. entities that
  have a real vocabulary URI to be anchored to).
- This is where the external-structure thesis is expected to show; internal
  structure only scores here to the extent its nodes carry / can be mapped to URIs.

Report precision, recall, F1 at both levels, per pipeline × per query_type.

---

## 3. The shared resolver (applied identically to every pipeline)

A single deterministic function `resolve(label) -> URI | None`, used the same way
for all pipelines. No per-case human judgement.

**Normalisation rules (fixed in advance):**
1. lowercase
2. strip surrounding whitespace/punctuation
3. singular/plural folding (naive: trailing -s handled via lexicon, not blind strip)
4. match against vocabulary `skos:prefLabel` AND `skos:altLabel`
   (altLabels exist precisely to capture synonyms — using them is not generosity,
   it is correct vocabulary use)
5. no fuzzy/edit-distance matching (keeps the resolver deterministic and auditable;
   a fuzzy match would itself become an unauditable step)

**Application per pipeline:**
- ontology-grounded: entities are already URIs → resolver is identity/no-op
  (still passed through for symmetry).
- graph-RAG: each retrieved node's **label** is passed through `resolve()`.
- plain-text: entities are extracted from the answer text with the SAME extractor
  for all such cases, then labels passed through `resolve()`.

**Mapping failure handling:** if a pipeline's retrieved label does not resolve to
any URI, that entity counts at concept-level (if its normalised label matches an
expected label) but NOT at grounding-level. This is intentional and symmetric:
an unanchorable entity is a real, measurable property, not discarded noise.

---

## 4. Per-entry fields to ADD to each gold query

Add these to each Q-entry so the scorer has an unambiguous target at both levels:

```yaml
expected_entities:
  - label: "whey"                       # used for concept-level match
    normalised_label: "whey"            # canonical form the resolver targets
    alt_labels: ["milk whey", "siero"]  # accepted concept-level variants
    uri: "http://aims.fao.org/aos/agrovoc/c_8376"
    mapping_status: "exact"             # -> counts at grounding-level
  - label: "scotta"
    normalised_label: "scotta"
    alt_labels: []
    uri: null
    mapping_status: "benchmark_local_extension"  # concept-level only
```

- `normalised_label` + `alt_labels` define what counts as a concept-level hit.
- `mapping_status: exact` is the flag that includes the entity in grounding-level.
- entries with `urn:ceff:` or unresolved concepts are concept-level only by design.

---

## 5. Cross-lingual note (already relevant to this corpus)

Queries are English; two source docs are Italian. `alt_labels` therefore should
include the Italian surface form where a pipeline might surface it (e.g. "siero"
for whey). AGROVOC altLabels are multilingual, so the resolver can map an Italian
label to the canonical URI — this is a genuine strength of vocabulary grounding
and should be allowed (and noted), not filtered out.

---

## 6. What to fix in advance vs report after

FIX BEFORE RUNNING (pre-registration, protects against outcome-tuning):
- normalisation rules (§3)
- the two-metric split (§2)
- alt_labels per entry (§4)
- mapping-failure handling (§3)

REPORT AFTER (these are results, not choices):
- concept-level P/R/F1 per pipeline
- grounding-level P/R/F1 per pipeline
- the gap between the two levels per pipeline (this gap IS the interoperability finding)

---

## 7. One-line rationale for the paper

> "Entities are evaluated at two levels — concept-level (normalised labels, a
> pipeline-agnostic retrieval measure) and grounding-level (resolved canonical
> URIs, an interoperability measure) — via a single shared resolver applied
> symmetrically to all pipelines, fixed prior to evaluation."
