"""LLM translator: instruction -> motion description -> reward code -> psi.

Providers
---------
anthropic : Claude via the official SDK (default model claude-opus-5-5).
openai    : OpenAI chat completions (the original L2R repo used GPT-4).
replay    : read previously stored responses from a JSONL file, so that every
            controller is evaluated on exactly the same rewards (proposal Sec. 3).

Each instruction is translated once, independently (fresh conversation), with
the fixed prompts in ``l2rvf/llm/prompts.py``.
"""

from __future__ import annotations

import json
import pathlib

from l2rvf.llm import prompts
from l2rvf.llm import reward_api

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-4o"}


def _user_message(task: str, instruction: str) -> str:
  return f"{instruction} {prompts.task_context(task)} Make sure to ignore irrelevant options."


class AnthropicLLM:
  def __init__(self, model: str):
    import anthropic

    self.anthropic = anthropic
    self.client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile
    self.model = model

  def chat(self, system_prompt: str, user: str) -> str:
    # Mirrors the original repo: the prompt is the first user turn.
    response = self.client.beta.messages.create(
        model=self.model,
        max_tokens=4000,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": "Understood. Please give me the task."},
            {"role": "user", "content": user},
        ],
    )
    if response.stop_reason == "refusal":
      raise RuntimeError(f"model refused: {response.stop_details}")
    return "".join(b.text for b in response.content if b.type == "text")


class OpenAILLM:
  def __init__(self, model: str):
    import openai

    self.client = openai.OpenAI()
    self.model = model

  def chat(self, system_prompt: str, user: str) -> str:
    r = self.client.chat.completions.create(
        model=self.model,
        temperature=0.3,
        messages=[
            {"role": "user", "content": system_prompt},
            {"role": "user", "content": user},
        ],
    )
    return r.choices[0].message.content


def translate(llm, task: str, instruction: str) -> dict:
  """Two-stage L2R translation. Returns raw responses, code and psi."""
  user = _user_message(task, instruction)
  thinker = llm.chat(prompts.prompt_thinker, user)
  description = reward_api.extract_description(thinker)
  coder = llm.chat(prompts.prompt_coder, description)
  return finalize(task, instruction, thinker, coder)


def finalize(task, instruction, thinker_response, coder_response, source=None):
  code = reward_api.process_code_block(coder_response)
  out = reward_api.execute(code)
  rec = {
      "task": task,
      "instruction": instruction,
      "thinker_response": thinker_response,
      "coder_response": coder_response,
      "code": code,
      "psi": out["psi"].tolist(),
      "calls": out["calls"],
      "duration": out["duration"],
  }
  if source:
    rec["source"] = source
  return rec


def load_replay(path: pathlib.Path) -> list[dict]:
  return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
