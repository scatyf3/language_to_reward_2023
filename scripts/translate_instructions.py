"""Translate every instruction in data/instructions.json once and store psi.

  # Real LLM calls (needs ANTHROPIC_API_KEY or OPENAI_API_KEY):
  python scripts/translate_instructions.py --provider anthropic
  # Re-execute stored raw responses (no network):
  python scripts/translate_instructions.py --provider replay --replay data/llm_responses.jsonl

Writes data/llm_responses.jsonl (raw responses + code) and data/psi_cache.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from l2rvf.arm import reward as R  # noqa: E402
from l2rvf.llm import translator as tr  # noqa: E402


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--provider", choices=["anthropic", "openai", "replay"], default="anthropic")
  ap.add_argument("--model", default=None)
  ap.add_argument("--replay", default=str(ROOT / "data" / "llm_responses.jsonl"))
  ap.add_argument("--out_responses", default=str(ROOT / "data" / "llm_responses.jsonl"))
  ap.add_argument("--out_psi", default=str(ROOT / "data" / "psi_cache.json"))
  args = ap.parse_args()

  instructions = json.loads((ROOT / "data" / "instructions.json").read_text(encoding="utf-8"))
  records = []
  if args.provider == "replay":
    for r in tr.load_replay(pathlib.Path(args.replay)):
      records.append(tr.finalize(r["task"], r["instruction"], r["thinker_response"],
                                 r["coder_response"], r.get("source")))
  else:
    model = args.model or tr.DEFAULT_MODELS[args.provider]
    llm = tr.AnthropicLLM(model) if args.provider == "anthropic" else tr.OpenAILLM(model)
    for task, items in instructions.items():
      for ins in items:
        rec = tr.translate(llm, task, ins)
        rec["source"] = f"{args.provider}:{model}"
        records.append(rec)
        print(f"[{task}] {ins}\n    -> {R.describe(__import__('numpy').array(rec['psi']))}")

  if args.provider != "replay" or args.out_responses != args.replay:
    with open(args.out_responses, "w", encoding="utf-8") as f:
      for r in records:
        f.write(json.dumps({k: r[k] for k in ("task", "instruction", "thinker_response",
                                              "coder_response", "source") if k in r}) + "\n")
  cache = {"psi_layout": R.__doc__.strip().splitlines()[5:], "entries": [
      {k: r[k] for k in ("task", "instruction", "psi", "calls", "source") if k in r}
      for r in records]}
  pathlib.Path(args.out_psi).write_text(json.dumps(cache, indent=1), encoding="utf-8")
  import numpy as np
  for r in records:
    print(f"[{r['task']}] {r['instruction'][:45]:45s} {R.describe(np.array(r['psi']))}")
  print(f"wrote {len(records)} entries to {args.out_psi}")


if __name__ == "__main__":
  main()
