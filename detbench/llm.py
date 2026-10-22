"""Calls Claude through the Claude Code CLI in headless mode (`claude -p`).

Running the unmodified Claude Code binary keeps usage on the user's own
Pro/Max subscription, which Anthropic's terms allow for Claude Code; the Agent
SDK would require an API key instead. Set backend: api in config.yaml to use
ANTHROPIC_API_KEY through the `anthropic` package.
"""
from __future__ import annotations

import json
import os
import re
import subprocess


class LLMError(RuntimeError):
    pass


class Claude:
    def __init__(self, model: str = "sonnet", backend: str = "cli", timeout: int = 600,
                 max_tokens: int = 4096):
        self.model = model
        self.backend = backend
        self.timeout = timeout
        self.max_tokens = max_tokens

    def complete(self, prompt: str, system: str = "") -> str:
        if self.backend == "api":
            return self._api(prompt, system)
        cmd = ["claude", "-p", "--output-format", "json", "--model", self.model,
               "--tools", "", "--no-session-persistence"]
        if system:
            cmd += ["--system-prompt", system]
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                              timeout=self.timeout, env=os.environ.copy())
        if proc.returncode != 0:
            raise LLMError(f"claude -p a échoué ({proc.returncode}) : {proc.stderr[-500:]}")
        data = json.loads(proc.stdout)
        if data.get("is_error"):
            raise LLMError(f"claude -p : {data.get('result')}")
        return data["result"].strip()

    def _api(self, prompt: str, system: str) -> str:
        import anthropic

        msg = anthropic.Anthropic().messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def complete_json(self, prompt: str, system: str = "") -> dict | list:
        out = self.complete(prompt, system)
        m = re.search(r"```(?:json)?\s*(.*?)```", out, re.S)
        payload = m.group(1) if m else out
        start = min([i for i in (payload.find("{"), payload.find("[")) if i >= 0], default=0)
        try:
            return json.loads(payload[start:])
        except json.JSONDecodeError as e:
            raise LLMError(f"Réponse JSON invalide : {out[:300]}") from e
