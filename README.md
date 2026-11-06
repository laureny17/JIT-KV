# JIT-KV

A simulator comparing when KV-cache data should move to decode GPU memory in
disaggregated LLM serving.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
./scripts/download_traces.sh
pytest
```

## Loading a trace

```python
from jitkv.trace import load_trace

# load_factor=2.0 replays the same requests with arrivals twice as fast.
requests = load_trace("data/conversation_trace.jsonl", load_factor=2.0)
```
