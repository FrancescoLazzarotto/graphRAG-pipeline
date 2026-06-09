# Gold Question Schema (Food Domain)

Use the following CSV columns in `evaluation/gold_questions_template.csv`:

- `question_id` (string): stable id, format `q_food_###`
- `question` (string): user-facing question text
- `canonical_answer` (string): primary expected answer
- `answer_variants` (JSON array of strings): acceptable paraphrases
- `expected_entities` (JSON array of strings): entities expected in retrieval
- `gold_triples` (JSON array of objects): each object has `subject`, `predicate`, `object`
- `question_type` (enum): one of `factoid`, `relation`, `path`, `multi_hop`
- `difficulty` (enum/string): typical values `easy`, `medium`, `hard`
- `notes` (string): optional annotation notes
