#!/bin/sh
# Downloads the public Mooncake traces into data/, pinned to a fixed commit.
set -eu

commit=3cca71daccf2a7afb8fe3f0295358f70e3a69fdb
base="https://raw.githubusercontent.com/kvcache-ai/Mooncake/$commit/FAST25-release/traces"
data_dir="$(dirname "$0")/../data"

mkdir -p "$data_dir"
for name in conversation_trace toolagent_trace synthetic_trace; do
    curl -fsSL "$base/$name.jsonl" -o "$data_dir/$name.jsonl"
    echo "downloaded data/$name.jsonl"
done
