import os
import json
import asyncio
import time
import base64
import mimetypes
import pathlib
import random  # Added: backoff jitter
from collections import deque
from typing import List, Tuple, Optional, Type, Literal, Any

from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import tiktoken
from tqdm import tqdm
import nest_asyncio

from model.model import LLM

nest_asyncio.apply()


# ---------- helpers ----------
def _b64_image_url(src: str) -> str:
    """Return http(s)/data URL as-is; convert local path to data:URL (base64)."""
    if src.startswith(("http://", "https://", "data:")):
        return src
    path = pathlib.Path(src).expanduser()
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from dict-like or object-like SDK payloads."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class APIModel(LLM):
    """OpenAI Responses API wrapper used by the public MedThinkVQA snapshot."""

    Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    Verbosity = Literal["low", "medium", "high"]
    DEFAULT_RPM_LIMIT = 30_000
    DEFAULT_TPM_LIMIT = 150_000_000
    DEFAULT_MAX_CONCURRENCY = 50
    # OpenAI prices in USD per 1M tokens.
    # Source:
    # - User-provided official pricing screenshot on 2026-03-17 for GPT-5.4 family.
    # - Existing fixed prices for other models are retained as before for deterministic cost logging.
    MODEL_PRICING_USD_PER_1M = {
        "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
        "gpt-5.4-pro": {"input": 30.00, "cached_input": None, "output": 180.00},
        "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
        "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
        "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
        "gpt-5.2-chat-latest": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5.2-codex": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
        "gpt-5.1-codex-max": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5.1-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-5.2-pro": {"input": 21.00, "cached_input": None, "output": 168.00},
        "gpt-5-pro": {"input": 15.00, "cached_input": None, "output": 120.00},
        "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
        "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
        "gpt-4o-2024-05-13": {"input": 5.00, "cached_input": None, "output": 15.00},
        "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
        "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
        "gpt-realtime-mini": {"input": 0.60, "cached_input": 0.06, "output": 2.40},
        "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
        "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
        "gpt-audio": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-audio-mini": {"input": 0.60, "cached_input": None, "output": 2.40},
        "gpt-4o-audio-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "gpt-4o-mini-audio-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
        "o1-pro": {"input": 150.00, "cached_input": None, "output": 600.00},
        "o3-pro": {"input": 20.00, "cached_input": None, "output": 80.00},
        "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
        "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
        "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
        "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
        "gpt-5.1-codex-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        "codex-mini-latest": {"input": 1.50, "cached_input": 0.375, "output": 6.00},
        "gpt-5-search-api": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
        "gpt-4o-mini-search-preview": {"input": 0.15, "cached_input": None, "output": 0.60},
        "gpt-4o-search-preview": {"input": 2.50, "cached_input": None, "output": 10.00},
        "computer-use-preview": {"input": 3.00, "cached_input": None, "output": 12.00},
        "gpt-image-1.5": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
        "chatgpt-image-latest": {"input": 5.00, "cached_input": 1.25, "output": 10.00},
        "gpt-image-1": {"input": 5.00, "cached_input": 1.25, "output": None},
        "gpt-image-1-mini": {"input": 2.00, "cached_input": 0.20, "output": None},
    }
    # Model-specific reasoning effort compatibility.
    # Source: OpenAI official docs (checked 2026-02-21):
    # - gpt-5.4 / gpt-5 / gpt-5-mini / gpt-5-nano: minimal|low|medium|high
    # - gpt-5.1: none|low|medium|high
    # - gpt-5.2: none|low|medium|high|xhigh
    # - gpt-5-pro: high
    # - gpt-5.2-pro: medium|high|xhigh
    _REASONING_EFFORTS_BY_PREFIX = (
        ("gpt-5.4-pro", ("high",)),
        ("gpt-5.4", ("minimal", "low", "medium", "high")),
        ("gpt-5.2-pro", ("medium", "high", "xhigh")),
        ("gpt-5-pro", ("high",)),
        ("gpt-5.2", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.1", ("none", "low", "medium", "high")),
        ("gpt-5-mini", ("minimal", "low", "medium", "high")),
        ("gpt-5-nano", ("minimal", "low", "medium", "high")),
        ("gpt-5", ("minimal", "low", "medium", "high")),
    )
    _UNSET = object()

    # ---------------------------- init ----------------------------
    def __init__(
        self,
        model_name: str,
        temperature: float = 1.0,
        max_concurrency: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        reasoning_effort: Optional[Effort] = None,
        verbosity: Optional[Verbosity] = None,
        print_cost: bool = True,             # Print per-generate cost summary
        set_max_tokens: bool = True,         # Whether to explicitly send max_tokens / max_output_tokens
        request_timeout_s: Optional[float] = None,  # Per-request timeout; None keeps SDK default
    ):
        super().__init__(model_name=model_name, temperature=temperature)
        load_dotenv()

        self.print_cost = bool(print_cost)
        self.set_max_tokens = bool(set_max_tokens)
        self.request_timeout_s = (
            float(request_timeout_s) if request_timeout_s is not None else None
        )
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive when provided.")
        self._compat_warnings_emitted = set()

        # Simple rate limit state
        self._req_times, self._token_times = deque(), deque()
        self._throttle_lock = asyncio.Lock()

        # Per-model caps (extend as needed)
        max_tokens_map = {
            "gpt-4o-mini": 16384,
            "gpt-4o": 16384,
            "gpt-o3-mini": 100000,
            "gpt-o4-mini": 100000,
            "gpt-4.1-mini": 32768,
        }
        self.max_tokens = max_tokens_map.get(self.model_name, 26_384)  # Per your requirement: do not modify
        self.max_concurrency = (
            int(max_concurrency) if max_concurrency is not None else self.DEFAULT_MAX_CONCURRENCY
        )
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        self.rpm_limit = int(rpm_limit) if rpm_limit is not None else self.DEFAULT_RPM_LIMIT
        self.tpm_limit = int(tpm_limit) if tpm_limit is not None else self.DEFAULT_TPM_LIMIT
        self.tokens_used = 0
        self.cached_input_tokens_used = 0
        self.reasoning_output_tokens_used = 0
        self.requests_made = 0
        self.total_cost_usd = 0.0
        self.costed_requests = 0
        self.uncosted_requests = 0

        # Reasoning/verbosity knobs for the OpenAI Responses API path.
        self.reasoning_effort_requested = self._reasoning_effort_norm(reasoning_effort)
        self.reasoning_effort = self.reasoning_effort_requested
        self.verbosity = verbosity
        if self.reasoning_effort and self.reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh"}:
            raise ValueError("reasoning_effort must be: none | minimal | low | medium | high | xhigh")
        if self.verbosity and self.verbosity not in {"low", "medium", "high"}:
            raise ValueError("verbosity must be: low | medium | high")
        self.reasoning_effort = self._resolve_reasoning_effort(self.reasoning_effort)

        print(
            f'{self.model_name} | max_out={self.max_tokens if self.set_max_tokens else "default"} '
            f'max_conc={self.max_concurrency} rpm={self.rpm_limit} '
            f'tpm={self.tpm_limit} effort={self.reasoning_effort} '
            f'effort_requested={self.reasoning_effort_requested} verbosity={self.verbosity} '
            f'timeout={self.request_timeout_s if self.request_timeout_s is not None else "default"} '
            f'channel=openai-responses'
        )

        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = AsyncOpenAI(api_key=self.api_key)
        if not hasattr(self.client, "responses"):
            raise RuntimeError(
                "Your openai package is too old and does not support Responses API. "
                "Run: pip install -U openai"
            )

        # Tokenizer (best-effort)
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            print(f"WARNING: tiktoken has no encoding for {self.model_name}; falling back to cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _model_name_norm(self) -> str:
        return str(self.model_name or "").strip().lower()

    def _reasoning_effort_norm(self, value: Any = _UNSET) -> Optional[str]:
        if value is self._UNSET:
            src = getattr(self, "reasoning_effort", None)
        else:
            src = value
        if src is None:
            return None
        try:
            s = str(src).strip().lower()
        except Exception:
            return None
        return s or None

    def _supported_reasoning_efforts(self) -> Optional[Tuple[str, ...]]:
        name = self._model_name_norm()
        for prefix, efforts in self._REASONING_EFFORTS_BY_PREFIX:
            if name.startswith(prefix):
                return efforts
        return None

    def _resolve_reasoning_effort(self, effort: Optional[str]) -> Optional[str]:
        """
        Resolve requested reasoning effort to a model-compatible value.
        - If unsupported and a close fallback exists, downgrade automatically.
        - Otherwise omit reasoning.effort and let model default apply.
        """
        norm = self._reasoning_effort_norm(effort)
        if norm is None:
            return None

        supported = self._supported_reasoning_efforts()
        if not supported:
            return norm
        if norm in supported:
            return norm

        fallback = None
        if norm == "none" and "minimal" in supported:
            fallback = "minimal"
        elif norm == "xhigh" and "high" in supported:
            fallback = "high"

        if fallback is not None:
            warn_key = f"reasoning-effort-fallback|{self.model_name}|{norm}|{fallback}"
            if warn_key not in self._compat_warnings_emitted:
                self._compat_warnings_emitted.add(warn_key)
                print(
                    f"WARNING: reasoning_effort={norm!r} is unsupported by {self.model_name}; "
                    f"using {fallback!r} instead. Supported: {', '.join(supported)}."
                )
            return fallback

        warn_key = f"reasoning-effort-drop|{self.model_name}|{norm}"
        if warn_key not in self._compat_warnings_emitted:
            self._compat_warnings_emitted.add(warn_key)
            print(
                f"WARNING: reasoning_effort={norm!r} is unsupported by {self.model_name}; "
                "omitting reasoning.effort so model default is used. "
                f"Supported: {', '.join(supported)}."
            )
        return None

    def _supports_sampling_params(self) -> bool:
        """
        Whether this model/config supports sampling params like:
        temperature/top_p/logprobs.
        """
        name = self._model_name_norm()
        if "gpt-5" not in name:
            return True

        # gpt-5 / gpt-5-mini / gpt-5-nano: no sampling params.
        # gpt-5.1 / gpt-5.2: sampling params are supported only when
        # reasoning_effort is none (or omitted, where default is none).
        # Codex variants are treated conservatively as unsupported here.
        if "codex" in name or "pro" in name:
            return False

        effort = self._reasoning_effort_norm()
        if name.startswith("gpt-5.1") or name.startswith("gpt-5.2"):
            effective_effort = effort if effort is not None else "none"
            return effective_effort == "none"
        return False

    def _sanitize_sampling_params(self, req: dict, endpoint: str) -> dict:
        if self._supports_sampling_params():
            return req
        removed = []
        for key in ("temperature", "top_p", "logprobs", "top_logprobs"):
            if key in req:
                req.pop(key, None)
                removed.append(key)
        if removed:
            warn_key = f"{endpoint}|{self.model_name}|{self._reasoning_effort_norm()}|{','.join(sorted(removed))}"
            if warn_key not in self._compat_warnings_emitted:
                self._compat_warnings_emitted.add(warn_key)
                print(
                    f"WARNING: {self.model_name} does not support {', '.join(sorted(removed))} "
                    "for the current configuration; these parameters were removed automatically."
                )
        return req

    @staticmethod
    def _as_int(x: Any) -> int:
        try:
            return int(x or 0)
        except Exception:
            return 0

    @staticmethod
    def _cost_component(tokens: int, rate_per_1m: Optional[float]) -> Optional[float]:
        if tokens <= 0:
            return 0.0
        if rate_per_1m is None:
            return None
        return (tokens / 1_000_000.0) * float(rate_per_1m)

    def _extract_usage_tokens(self, usage: Any) -> Tuple[int, int, int, int]:
        """
        Extract (input_tokens, output_tokens, cached_input_tokens, reasoning_output_tokens)
        from either Responses usage or Chat usage payloads.
        """
        in_tok = _obj_get(usage, "input_tokens", None)
        if in_tok is None:
            in_tok = _obj_get(usage, "prompt_tokens", 0)

        out_tok = _obj_get(usage, "output_tokens", None)
        if out_tok is None:
            out_tok = _obj_get(usage, "completion_tokens", 0)

        in_details = _obj_get(usage, "input_tokens_details", None)
        prompt_details = _obj_get(usage, "prompt_tokens_details", None)
        cached_tok = _obj_get(in_details, "cached_tokens", None)
        if cached_tok is None:
            cached_tok = _obj_get(prompt_details, "cached_tokens", 0)
        if cached_tok is None:
            cached_tok = _obj_get(usage, "cached_tokens", 0)

        out_details = _obj_get(usage, "output_tokens_details", None)
        out_details_legacy = _obj_get(usage, "output_token_details", None)
        completion_details = _obj_get(usage, "completion_tokens_details", None)
        reasoning_tok = _obj_get(out_details, "reasoning_tokens", None)
        if reasoning_tok is None:
            reasoning_tok = _obj_get(out_details_legacy, "reasoning_tokens", None)
        if reasoning_tok is None:
            reasoning_tok = _obj_get(completion_details, "reasoning_tokens", 0)

        in_i = self._as_int(in_tok)
        out_i = self._as_int(out_tok)
        cached_i = self._as_int(cached_tok)
        reasoning_i = self._as_int(reasoning_tok)

        if cached_i < 0:
            cached_i = 0
        if cached_i > in_i:
            cached_i = in_i
        if reasoning_i < 0:
            reasoning_i = 0
        if reasoning_i > out_i:
            reasoning_i = out_i
        return in_i, out_i, cached_i, reasoning_i

    def _estimate_cost(
        self, in_tok: int, out_tok: int, cached_in_tok: int, reasoning_out_tok: int = 0
    ) -> Optional[dict]:
        """
        Estimate request cost from token usage and static model prices.
        Returns None when model pricing is not configured.
        """
        pricing = self.MODEL_PRICING_USD_PER_1M.get(self.model_name)
        if pricing is None:
            return None

        in_tok_i = max(0, self._as_int(in_tok))
        out_tok_i = max(0, self._as_int(out_tok))
        cached_i = min(max(0, self._as_int(cached_in_tok)), in_tok_i)
        uncached_i = max(0, in_tok_i - cached_i)
        reasoning_i = min(max(0, self._as_int(reasoning_out_tok)), out_tok_i)
        visible_i = max(0, out_tok_i - reasoning_i)

        input_rate = pricing.get("input")
        cached_rate_raw = pricing.get("cached_input")
        output_rate = pricing.get("output")

        input_cost = self._cost_component(uncached_i, input_rate)
        if cached_i <= 0:
            cached_cost = 0.0
        elif cached_rate_raw is None:
            # Pricing table marks cached input as "-" for this model; do not guess.
            cached_cost = None
        else:
            cached_cost = self._cost_component(cached_i, cached_rate_raw)
        output_cost = self._cost_component(out_tok_i, output_rate)

        if input_cost is None or cached_cost is None or output_cost is None:
            total_cost = None
        else:
            total_cost = input_cost + cached_cost + output_cost

        return {
            "input_tokens": in_tok_i,
            "cached_input_tokens": cached_i,
            "uncached_input_tokens": uncached_i,
            "output_tokens": out_tok_i,
            "reasoning_output_tokens": reasoning_i,
            "visible_output_tokens": visible_i,
            "input_cost_usd": input_cost,
            "cached_input_cost_usd": cached_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "cached_input_pricing_missing": (cached_i > 0 and cached_rate_raw is None),
        }

    # ---------------------------- rate throttle ----------------------------
    async def _throttle(self, tokens_est: int):
        async with self._throttle_lock:
            now = time.monotonic()
            while self._req_times and now - self._req_times[0] > 60:
                self._req_times.popleft()
            while self._token_times and now - self._token_times[0][0] > 60:
                self._token_times.popleft()

            rpm_wait = 0
            if self.rpm_limit and len(self._req_times) >= self.rpm_limit:
                rpm_wait = (self._req_times[0] + 60) - now

            tpm_wait = 0
            if self.tpm_limit:
                used = sum(t for _, t in self._token_times)
                need = used + tokens_est - self.tpm_limit
                if need > 0:
                    freed = 0
                    for ts, tk in self._token_times:
                        freed += tk
                        if freed >= need:
                            tpm_wait = (ts + 60) - now
                            break
                    else:
                        tpm_wait = (self._token_times[-1][0] + 60) - now

        wait = max(rpm_wait, tpm_wait, 0)
        if wait > 0:
            await asyncio.sleep(wait)
            return await self._throttle(tokens_est)

        async with self._throttle_lock:
            self._req_times.append(now)
            entry = [now, tokens_est]
            self._token_times.append(entry)
            return entry

    # ---------- internal builders ----------
    def _build_responses_input(
        self, system_msg: str, user_msg: str, images: Optional[List[str]]
    ) -> list:
        """Build inputs for Responses API (text+images)."""
        content = []
        if user_msg:
            content.append({"type": "input_text", "text": user_msg})
        if images:
            content.extend({"type": "input_image", "image_url": _b64_image_url(x)} for x in images)

        inputs = []
        if system_msg:
            inputs.append({"role": "system", "content": system_msg})
        inputs.append({"role": "user", "content": content if content else user_msg})
        return inputs

    async def _responses_call_with_retries(
        self,
        req: dict,
        schema: Optional[Type[BaseModel]] = None,
        max_retries: int = 5,
    ):
        """
        Robust Responses API call with:
          1) original request
          2) one-time compatibility downgrade (drop text/reasoning/temperature)
          3) exponential backoff retries
        """
        if max_retries <= 0:
            raise ValueError("max_retries must be a positive integer.")

        last_err: Optional[Exception] = None
        call_req = dict(req)
        use_parse = schema is not None
        downgraded = False

        for attempt in range(max_retries):
            try:
                if use_parse:
                    return await self._await_with_timeout(
                        self.client.responses.parse(
                            **call_req,  # type: ignore[arg-type]
                            text_format=schema,
                        )
                    )
                return await self._await_with_timeout(
                    self.client.responses.create(**call_req)  # type: ignore[arg-type]
                )
            except Exception as e:
                last_err = e

                # One-time compatibility downgrade first (no backoff yet).
                if not downgraded:
                    fallback_req = dict(call_req)
                    dropped_reasoning = ("reasoning" in fallback_req)
                    removed_any = False
                    for k in ("text", "reasoning", "temperature"):
                        if k in fallback_req:
                            fallback_req.pop(k, None)
                            removed_any = True
                    if removed_any:
                        call_req = fallback_req
                        downgraded = True
                        if dropped_reasoning and (self.reasoning_effort is not None):
                            print(
                                "WARNING: initial Responses request failed with "
                                f"reasoning_effort={self.reasoning_effort}; retrying without reasoning. "
                                "Model-side default reasoning behavior may apply. "
                                f"error={type(e).__name__}: {e}"
                            )
                        await asyncio.sleep(0.2)
                        continue

                # Exponential backoff + jitter (cap ~10s)
                if attempt < (max_retries - 1):
                    backoff = min(2 ** attempt, 10) + random.random()
                    await asyncio.sleep(backoff)
                    continue

                raise

        raise last_err if last_err is not None else RuntimeError(
            "Responses call failed without an exception."
        )

    async def _await_with_timeout(self, coro):
        if self.request_timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=self.request_timeout_s)

    # ---------------------------- single request ----------------------------
    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[str]] = None,
        schema: Optional[Type[BaseModel]] = None
    ) -> Tuple[Any, int, int]:

        # 1) throttle
        est_tokens = (self.tokens_used / self.requests_made) * 1.5 if self.requests_made else self.max_tokens
        reserve = await self._throttle(est_tokens)

        # 2) OpenAI Responses API
        inputs = self._build_responses_input(system_msg, user_msg, images)

        base_req = {
            "model": self.model_name,
            "input": inputs,
            "temperature": self.temperature,
        }
        if self.set_max_tokens:
            base_req["max_output_tokens"] = self.max_tokens
        base_req = self._sanitize_sampling_params(base_req, endpoint="responses")
        if self.reasoning_effort:
            base_req["reasoning"] = {"effort": self.reasoning_effort}

        use_parse = schema is not None
        if (self.verbosity is not None) and (not use_parse):
            base_req["text"] = {"verbosity": self.verbosity}

        resp = await self._responses_call_with_retries(
            req=base_req,
            schema=schema,
            max_retries=5,
        )

        usage = getattr(resp, "usage", None) or {}
        in_tok, out_tok, cached_in_tok, reasoning_out_tok = self._extract_usage_tokens(usage)

        parsed = getattr(resp, "output_parsed", None)
        if parsed is not None:
            if isinstance(parsed, BaseModel):
                result = parsed.model_dump()
            else:
                result = parsed
        else:
            txt = getattr(resp, "output_text", None)
            if txt is None:
                try:
                    parts = []
                    for o in getattr(resp, "output", []) or []:
                        if getattr(o, "type", "") == "message":
                            for c in getattr(o, "content", []) or []:
                                if getattr(c, "type", "") in ("output_text", "input_text"):
                                    parts.append(getattr(c, "text", "") or "")
                    txt = "".join(parts)
                except Exception:
                    txt = ""
            try:
                obj = json.loads(txt) if txt else {}
            except Exception:
                obj = txt or {}
            if (schema is not None) and (parsed is None):
                try:
                    validated = schema.model_validate(obj)
                    result = validated.model_dump()
                except Exception:
                    result = obj
            else:
                result = obj

        # 4) bookkeeping for throttle
        cost_info = self._estimate_cost(
            int(in_tok or 0),
            int(out_tok or 0),
            int(cached_in_tok or 0),
            int(reasoning_out_tok or 0),
        )
        async with self._throttle_lock:
            total = int(in_tok or 0) + int(out_tok or 0)
            self.tokens_used += total
            self.cached_input_tokens_used += int(cached_in_tok or 0)
            self.reasoning_output_tokens_used += int(reasoning_out_tok or 0)
            self.requests_made += 1
            if cost_info is not None and cost_info.get("total_cost_usd") is not None:
                self.total_cost_usd += float(cost_info["total_cost_usd"])
                self.costed_requests += 1
            else:
                self.uncosted_requests += 1
            reserve[1] = total

        return result, int(in_tok or 0), int(out_tok or 0)

    # ---------------------------- batch ----------------------------
    async def generate_async(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[Any, int, int]]:

        print_cost: bool = bool(kwargs.pop("print_cost", self.print_cost))
        max_concurrency = kwargs.pop("max_concurrency", None)
        if max_concurrency is None:
            effective_max_concurrency = self.max_concurrency
        else:
            effective_max_concurrency = int(max_concurrency)
            if effective_max_concurrency <= 0:
                raise ValueError("max_concurrency must be a positive integer.")
        semaphore = asyncio.Semaphore(effective_max_concurrency)

        async with self._throttle_lock:
            base_cached = int(self.cached_input_tokens_used)
            base_reasoning = int(self.reasoning_output_tokens_used)
            base_cost = float(self.total_cost_usd)
            base_costed = int(self.costed_requests)
            base_uncosted = int(self.uncosted_requests)

        async def _wrap(idx: int, sys_msg: str, user_msg: str, img_list: Optional[List[str]]):
            try:
                async with semaphore:
                    out = await self._generate_single(sys_msg, user_msg, img_list, schema)
                return idx, out
            except Exception as e:
                # Fallback: a single failure won't crash the whole batch
                placeholder = ({"__error__": f"{type(e).__name__}: {e}"}, 0, 0)
                return idx, placeholder

        tasks = []
        for i, tup in enumerate(prompts):
            sys_msg, user_msg = tup[0], tup[1]
            img_list = tup[2] if len(tup) > 2 else None
            tasks.append(asyncio.create_task(_wrap(i, sys_msg, user_msg, img_list)))

        final_results: List[Tuple[Any, int, int]]
        if not show_progress:
            done = await asyncio.gather(*tasks)
            done.sort(key=lambda x: x[0])
            final_results = [r for _, r in done]
        else:
            results: List[Optional[Tuple[Any, int, int]]] = [None] * len(prompts)
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{self.model_name} generating..."):
                idx, res = await fut
                results[idx] = res
            final_results = [
                r if r is not None else ({"__error__": "RuntimeError: missing async result"}, 0, 0)
                for r in results
            ]

        if print_cost:
            call_input_tok = sum(int((r[1] if len(r) > 1 else 0) or 0) for r in final_results)
            call_output_tok = sum(int((r[2] if len(r) > 2 else 0) or 0) for r in final_results)
            call_failed = sum(
                1 for r in final_results
                if isinstance(r, (list, tuple))
                and len(r) > 0
                and isinstance(r[0], dict)
                and "__error__" in r[0]
            )

            async with self._throttle_lock:
                call_cached_tok = int(self.cached_input_tokens_used) - base_cached
                call_reasoning_tok = int(self.reasoning_output_tokens_used) - base_reasoning
                call_cost_usd = float(self.total_cost_usd) - base_cost
                call_costed = int(self.costed_requests) - base_costed
                call_uncosted = int(self.uncosted_requests) - base_uncosted

            call_cached_tok = max(0, call_cached_tok)
            call_uncached_tok = max(0, call_input_tok - call_cached_tok)
            call_reasoning_tok = max(0, min(call_reasoning_tok, call_output_tok))
            call_visible_out_tok = max(0, call_output_tok - call_reasoning_tok)

            summary = (
                f"[COST] model={self.model_name} n={len(final_results)} "
                f"input={call_input_tok} (cached={call_cached_tok}, uncached={call_uncached_tok}) "
                f"output={call_output_tok} (reasoning={call_reasoning_tok}, visible={call_visible_out_tok}) "
                f"est_cost=${call_cost_usd:.6f}"
            )
            if call_costed < len(final_results):
                unpriced_count = max(0, len(final_results) - call_costed)
                if call_uncosted > 0:
                    unpriced_count = max(unpriced_count, call_uncosted)
                summary += (
                    f" | priced={call_costed} unpriced={unpriced_count}"
                )
            if call_failed > 0:
                summary += f" | failed={call_failed}"
            print(summary)

        return final_results

    # ---------------------------- sync wrapper ----------------------------
    def generate(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[Any, int, int]]:
        """
        Jupyter-safe sync wrapper: works both in scripts and running event loops.
        """
        coro = self.generate_async(prompts, schema, show_progress, **kwargs)
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(coro)  # type: ignore[no-any-return]
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(coro)  # type: ignore[no-any-return]
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
