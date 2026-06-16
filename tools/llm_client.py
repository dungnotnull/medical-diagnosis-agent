"""Unified LLM client: Claude (primary), OpenAI (fallback), Ollama (offline/privacy)."""
from __future__ import annotations

import logging
import os
import time
from typing import Generator, Optional

logger = logging.getLogger(__name__)

COST_PER_1K = {
    "claude-opus-4-8":      {"input": 0.015, "output": 0.075},
    "claude-sonnet-4-6":    {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5":     {"input": 0.00025, "output": 0.00125},
    "gpt-4o":               {"input": 0.005, "output": 0.015},
    "gpt-4o-mini":          {"input": 0.00015, "output": 0.0006},
    "ollama/llama3":        {"input": 0.0, "output": 0.0},
    "ollama/mistral":       {"input": 0.0, "output": 0.0},
}


class LLMResult:
    def __init__(self, text: str, provider: str, model: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        self.text = text
        self.provider = provider
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def cost_usd(self) -> float:
        rates = COST_PER_1K.get(self.model, {"input": 0.0, "output": 0.0})
        return (self.prompt_tokens / 1000 * rates["input"]) + (self.completion_tokens / 1000 * rates["output"])


class UnifiedLLMClient:
    def __init__(self, memory_manager=None):
        self._memory = memory_manager
        self._last_provider = "none"
        self._privacy_mode = os.getenv("PRIVACY_MODE", "").lower() in ("1", "true", "yes")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._openai_key = os.getenv("OPENAI_API_KEY", "")
        self._ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._claude_model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
        self._openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

    def _build_provider_chain(self) -> list[str]:
        if self._privacy_mode:
            return ["ollama"]
        chain = []
        if self._anthropic_key:
            chain.append("claude")
        if self._openai_key:
            chain.append("openai")
        chain.append("ollama")
        return chain

    def complete(self, prompt: str, max_tokens: int = 800, temperature: float = 0.1) -> str:
        result = self._complete_with_result(prompt, max_tokens, temperature)
        return result.text

    def _complete_with_result(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        chain = self._build_provider_chain()
        last_error = None
        for provider in chain:
            for attempt in range(3):
                try:
                    if provider == "claude":
                        result = self._call_claude(prompt, max_tokens, temperature)
                    elif provider == "openai":
                        result = self._call_openai(prompt, max_tokens, temperature)
                    else:
                        result = self._call_ollama(prompt, max_tokens, temperature)
                    self._last_provider = provider
                    self._log_cost(result)
                    return result
                except Exception as e:
                    last_error = e
                    wait = 2 ** attempt
                    logger.warning("Provider %s attempt %d failed: %s — retrying in %ds", provider, attempt + 1, e, wait)
                    time.sleep(wait)
        logger.error("All LLM providers failed: %s", last_error)
        return LLMResult(
            text="[LLM unavailable — using heuristic fallback]",
            provider="fallback",
            model="none",
        )

    def _call_claude(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        import anthropic
        client = anthropic.Anthropic(api_key=self._anthropic_key)
        response = client.messages.create(
            model=self._claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return LLMResult(
            text=text,
            provider="claude",
            model=self._claude_model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )

    def _call_openai(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        import openai
        client = openai.OpenAI(api_key=self._openai_key)
        response = client.chat.completions.create(
            model=self._openai_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResult(
            text=text,
            provider="openai",
            model=self._openai_model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def _call_ollama(self, prompt: str, max_tokens: int, temperature: float) -> LLMResult:
        import urllib.request, json
        payload = json.dumps({
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(
            f"{self._ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        text = data.get("response", "")
        return LLMResult(
            text=text,
            provider="ollama",
            model=f"ollama/{self._ollama_model}",
        )

    def stream(self, prompt: str, max_tokens: int = 800, temperature: float = 0.1) -> Generator[str, None, None]:
        result = self.complete(prompt, max_tokens, temperature)
        for word in result.split(" "):
            yield word + " "

    def complete_sync(self, prompt: str, max_tokens: int = 800, temperature: float = 0.1) -> str:
        return self.complete(prompt, max_tokens, temperature)

    def _log_cost(self, result: LLMResult) -> None:
        if self._memory and result.cost_usd > 0:
            try:
                self._memory.log_llm_cost(
                    provider=result.provider,
                    model=result.model,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    cost_usd=result.cost_usd,
                    use_case="medical_diagnosis",
                )
            except Exception:
                pass
