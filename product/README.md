# product/

The thing a person actually uses. Two surfaces over one engine.

| File | What it is |
|---|---|
| `app.py` | Browser UI (Streamlit): multiple chats, sources in an expander, model selector |
| `console.py` | Terminal REPL: same answers, no tunnel needed |
| `config.py` | The settings both of them run on, and how they reach the graph and the models |

## Running it

```bash
bash scripts/serving/start_demo.sh              # encoder + generator + UI, with a preflight
bash scripts/serving/start_demo.sh --list       # which models can be served
bash scripts/serving/stop_demo.sh               # everything down
```

Or one surface at a time, against servers that are already up:

```bash
conda run -n graphllm streamlit run product/app.py
conda run -n graphllm python product/console.py --strategy hybrid
```

## Where the line is

`src/graphrag/` is the **engine**: retrieval, the agent state machine, the LLM
layer. It is what the thesis measured, and changing its defaults makes future
runs incomparable with the campaigns already reported.

`product/` is **how the engine is presented**: which settings the demo runs on,
what the answers look like, how the whole thing starts. Change demo behaviour
here, never in `graphrag.config` or `graphrag.strategies`.

Every setting in `config.py` is an environment variable with the value the demo
ships with, so nothing needs editing to try something. The full list, with each
default, is in
[../docs/configuration.md](../docs/configuration.md#demo-settings):

```bash
DEMO_STRATEGY=default DEMO_COMPLEXITY=medium \
  conda run -n graphllm streamlit run product/app.py
```

The demos do **not** enable the vector channel, so a demo answer is not a
retrieval measurement. Read them as a presentation of the engine, never as a
data point.

`config.py` is also where the graph connection lives. The primary graph is a
hosted instance that suspends itself after three idle days; when it does not
answer, `DEMO_NEO4J_FALLBACK_URL` takes over so a booked session does not die
because a database went to sleep.

## Appearance

`.streamlit/config.toml` at the repository root (Streamlit reads it from the
working directory, not from here) holds a restrained base palette. It is a
starting point, not a finished identity — there is no logo, no custom CSS and
no typography work yet.
