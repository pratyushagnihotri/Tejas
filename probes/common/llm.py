"""Thin async client for an OpenAI-compatible server (vLLM on the DGX, Ollama locally).

Every call is counted (calls, prompt/completion tokens) so the probes can report
the logical work next to the measured energy, as GreenQuery does.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import List, Optional

import httpx
from openai import AsyncOpenAI

YES_NO_SYSTEM = "You are a precise classifier. Answer with exactly one word: yes or no."


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0  # reported by vLLM when prefix caching hits

    def add(self, u):
        self.calls += 1
        if u is None:
            return
        self.prompt_tokens += u.prompt_tokens or 0
        self.completion_tokens += u.completion_tokens or 0
        det = getattr(u, "prompt_tokens_details", None)
        if det is not None and getattr(det, "cached_tokens", None):
            self.cached_prompt_tokens += det.cached_tokens

    def to_dict(self):
        return self.__dict__.copy()


@dataclass
class LLM:
    base_url: str
    model: str
    max_concurrency: int = 64
    api_key: str = "EMPTY"
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self):
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600, max_retries=2)
        self.sem = asyncio.Semaphore(self.max_concurrency)

    def reset_usage(self):
        self.usage = Usage()

    async def yes_no(self, prompt: str) -> bool:
        async with self.sem:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": YES_NO_SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=2,
                temperature=0.0,
            )
        self.usage.add(r.usage)
        txt = (r.choices[0].message.content or "").strip().lower()
        return txt.startswith("y")

    async def yes_no_prob(self, prompt: str) -> dict:
        """Answer plus P(yes) from the first token's log-probabilities (for cascades)."""
        import math

        async with self.sem:
            r = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": YES_NO_SYSTEM}, {"role": "user", "content": prompt}],
                max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=10,
            )
        self.usage.add(r.usage)
        ch = r.choices[0]
        py = pn = 0.0
        try:
            for t in ch.logprobs.content[0].top_logprobs:
                tok = t.token.strip().lower()
                if tok.startswith("y"):
                    py += math.exp(t.logprob)
                elif tok.startswith("n"):
                    pn += math.exp(t.logprob)
        except Exception:
            pass
        ans = (ch.message.content or "").strip().lower().startswith("y")
        p_yes = py / (py + pn) if py + pn > 0 else float(ans)
        return {"answer": ans, "p_yes": p_yes,
                "prompt_tokens": r.usage.prompt_tokens if r.usage else None}

    async def yes_no_many(self, prompts: List[str]) -> List[bool]:
        return await asyncio.gather(*(self.yes_no(p) for p in prompts))

    async def complete(self, prompt: str, max_tokens: int = 64, system: Optional[str] = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        async with self.sem:
            r = await self.client.chat.completions.create(
                model=self.model, messages=msgs, max_tokens=max_tokens, temperature=0.0
            )
        self.usage.add(r.usage)
        return (r.choices[0].message.content or "").strip()

    def reset_prefix_cache(self) -> bool:
        """vLLM: POST /reset_prefix_cache (needs VLLM_SERVER_DEV_MODE=1). No-op elsewhere."""
        root = self.base_url.rstrip("/").removesuffix("/v1")
        try:
            r = httpx.post(f"{root}/reset_prefix_cache", timeout=30)
            return r.status_code == 200
        except Exception:
            return False


def from_env(model: Optional[str] = None, concurrency: Optional[int] = None) -> LLM:
    return LLM(
        base_url=os.environ.get("PROBE_BASE_URL", "http://localhost:8000/v1"),
        model=model or os.environ.get("PROBE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        max_concurrency=concurrency or int(os.environ.get("PROBE_CONCURRENCY", "64")),
    )
