from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import pathlib
import random
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx
import nest_asyncio
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm

from model.model import LLM

nest_asyncio.apply()


def _guess_media_type(src: str) -> str:
    base = str(src).split("?", 1)[0].split("#", 1)[0]
    return mimetypes.guess_type(base)[0] or "image/jpeg"


def _sniff_media_type(raw: bytes, fallback: str = "image/jpeg") -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith(b"BM"):
        return "image/bmp"
    if raw.startswith((b"II*\x00", b"MM\x00*")):
        return "image/tiff"
    return fallback


def _extract_text_from_message(resp: Any) -> str:
    texts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", "") == "text":
            t = getattr(block, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
    return "".join(texts)


class ClaudeModel(LLM):
    """Anthropic Claude Messages API wrapper."""

    DEFAULT_MAX_CONCURRENCY = 32
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_MAX_RETRIES = 5

    # Anthropic standard pricing (USD / 1M tokens), checked 2026-03-01.
    MODEL_PRICING_USD_PER_1M = {
        "claude-opus-4-6": {"input": 5.0, "output": 25.0},
        "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
        "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    }

    # Map existing project effort labels to Claude effort labels.
    _EFFORT_MAP = {
        "none": "low",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "max",
        "max": "max",
    }

    _SUPPORTED_EFFORTS_BY_PREFIX = (
        ("claude-opus-4-6", ("low", "medium", "high", "max")),
        ("claude-sonnet-4-6", ("low", "medium", "high")),
        ("claude-haiku-4-5", ("low", "medium", "high")),
    )

    _BUDGET_BY_EFFORT = {
        "low": 1024,
        "medium": 2048,
        "high": 4096,
        "max": 8192,
    }

    def __init__(
        self,
        model_name: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_concurrency: Optional[int] = None,
        reasoning_effort: Optional[str] = "high",
        thinking_mode: str = "adaptive",
        max_retries: int = DEFAULT_MAX_RETRIES,
        print_cost: bool = True,
    ) -> None:
        base_model_name = str(model_name or "").strip()
        if "/" not in base_model_name:
            base_model_name = f"Anthropic/{base_model_name}"
        super().__init__(model_name=base_model_name, temperature=temperature)
        load_dotenv()

        self.api_key = os.getenv("CLAUDE_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("CLAUDE_KEY or ANTHROPIC_API_KEY is not set.")

        self.client = AsyncAnthropic(api_key=self.api_key, max_retries=0)
        self.max_tokens = int(max_tokens)
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")

        self.max_concurrency = (
            int(max_concurrency) if max_concurrency is not None else self.DEFAULT_MAX_CONCURRENCY
        )
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")

        self.reasoning_effort_requested = self._normalize_effort(reasoning_effort)
        self.reasoning_effort = self._resolve_effort(self.reasoning_effort_requested)
        self.thinking_mode = str(thinking_mode or "adaptive").strip().lower()
        if self.thinking_mode not in {"adaptive", "enabled", "disabled"}:
            raise ValueError("thinking_mode must be one of: adaptive | enabled | disabled")

        self.max_retries = max(1, int(max_retries))
        self.print_cost = bool(print_cost)

        self.tokens_used = 0
        self.cached_input_tokens_used = 0
        self.reasoning_output_tokens_used = 0
        self.requests_made = 0
        self.total_cost_usd = 0.0
        self.costed_requests = 0
        self.uncosted_requests = 0
        self._stats_lock = asyncio.Lock()

        print(
            f"{self.model_name} | max_out={self.max_tokens} max_conc={self.max_concurrency} "
            f"thinking_mode={self.thinking_mode} effort={self.reasoning_effort} "
            f"effort_requested={self.reasoning_effort_requested}"
        )

    def _model_name_norm(self) -> str:
        return str(self.model_name or "").strip().lower()

    @staticmethod
    def _as_int(v: Any) -> int:
        try:
            return int(v or 0)
        except Exception:
            return 0

    def _normalize_effort(self, effort: Optional[str]) -> Optional[str]:
        if effort is None:
            return None
        s = str(effort).strip().lower()
        if not s:
            return None
        return self._EFFORT_MAP.get(s, s)

    def _supported_efforts(self) -> Optional[Tuple[str, ...]]:
        name = self._model_name_norm()
        for prefix, efforts in self._SUPPORTED_EFFORTS_BY_PREFIX:
            if name.startswith(prefix):
                return efforts
        return None

    def _resolve_effort(self, effort: Optional[str]) -> Optional[str]:
        if effort is None:
            return None
        supported = self._supported_efforts()
        if not supported:
            return effort
        if effort in supported:
            return effort
        # Conservative fallback for unsupported effort levels.
        if effort == "max" and "high" in supported:
            print(
                f"WARNING: effort={effort!r} is unsupported by {self.model_name}; using 'high' instead."
            )
            return "high"
        if "medium" in supported:
            print(
                f"WARNING: effort={effort!r} is unsupported by {self.model_name}; using 'medium' instead."
            )
            return "medium"
        return supported[-1]

    async def _image_block_from_src(self, src: str) -> Dict[str, Any]:
        src = str(src or "").strip()
        if not src:
            raise ValueError("Empty image source.")

        media_type_guess = _guess_media_type(src)
        raw: bytes
        if src.startswith(("http://", "https://")):
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(src)
                resp.raise_for_status()
                raw = resp.content
        elif src.startswith("data:"):
            head, sep, payload = src.partition(",")
            if not sep:
                raise ValueError("Invalid data URL image.")
            if ";base64" not in head:
                raise ValueError("Only base64 data URLs are supported for images.")
            media_type_guess = head[5:].split(";", 1)[0] or media_type_guess
            raw = base64.b64decode(payload, validate=True)
        else:
            path = pathlib.Path(src).expanduser()
            if not path.is_absolute():
                path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            raw = path.read_bytes()

        media_type = _sniff_media_type(raw, fallback=media_type_guess)
        b64 = base64.b64encode(raw).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        }

    def _json_only_suffix(self) -> str:
        return (
            "\n\nFormat:\n"
            "Return ONLY a single-line minified JSON object with keys exactly `answer` and `explanation`.\n"
            "Use exactly one uppercase letter for `answer`, chosen from the given Options.\n"
            'Example: {"answer":"B","explanation":"Key imaging features suggest ..."}'
        )

    async def _messages_create_with_retries(self, req: Dict[str, Any]) -> Any:
        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return await self.client.messages.create(**req)
            except Exception as exc:
                last_err = exc
                if attempt >= self.max_retries - 1:
                    raise
                backoff = min(2**attempt, 10) + random.random()
                await asyncio.sleep(backoff)
        raise RuntimeError(f"Claude request failed: {last_err}")

    def _thinking_attempts(self, base_req: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        attempts: List[Tuple[str, Dict[str, Any]]] = []

        if self.thinking_mode == "disabled":
            attempts.append(("disabled", dict(base_req)))
            return attempts

        if self.thinking_mode == "adaptive":
            # Recommended path for Claude 4.6 family.
            req_adaptive = dict(base_req)
            req_adaptive["temperature"] = 1.0
            extra_body: Dict[str, Any] = {"thinking": {"type": "adaptive"}}
            if self.reasoning_effort is not None:
                extra_body["output_config"] = {"effort": self.reasoning_effort}
            req_adaptive["extra_body"] = extra_body
            attempts.append(("adaptive", req_adaptive))

        # Compatibility fallback for SDK/API combinations that only support enabled/disabled.
        if self.max_tokens > 1024:
            req_enabled = dict(base_req)
            req_enabled["temperature"] = 1.0
            budget = self._BUDGET_BY_EFFORT.get(self.reasoning_effort or "high", 4096)
            budget = max(1024, min(budget, self.max_tokens - 1))
            req_enabled["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
            attempts.append(("enabled", req_enabled))

        # Final fallback without explicit thinking config.
        attempts.append(("default", dict(base_req)))
        return attempts

    def _extract_usage_tokens(self, usage: Any) -> Tuple[int, int, int, int]:
        input_tokens = self._as_int(getattr(usage, "input_tokens", 0))
        output_tokens = self._as_int(getattr(usage, "output_tokens", 0))
        cache_read = self._as_int(getattr(usage, "cache_read_input_tokens", 0))
        cache_create = self._as_int(getattr(usage, "cache_creation_input_tokens", 0))
        cached = max(0, min(input_tokens, cache_read + cache_create))
        return input_tokens, output_tokens, cached, 0

    def _pricing_key(self) -> Optional[str]:
        name = self._model_name_norm()
        for k in self.MODEL_PRICING_USD_PER_1M:
            if name.startswith(k):
                return k
        return None

    def _estimate_cost(
        self, in_tok: int, out_tok: int, cached_in_tok: int, reasoning_out_tok: int = 0
    ) -> Optional[dict]:
        key = self._pricing_key()
        if key is None:
            return None
        pricing = self.MODEL_PRICING_USD_PER_1M[key]
        in_tok_i = max(0, self._as_int(in_tok))
        out_tok_i = max(0, self._as_int(out_tok))
        cached_i = max(0, min(self._as_int(cached_in_tok), in_tok_i))
        uncached_i = max(0, in_tok_i - cached_i)
        reasoning_i = max(0, min(self._as_int(reasoning_out_tok), out_tok_i))
        visible_i = max(0, out_tok_i - reasoning_i)

        input_cost = (in_tok_i / 1_000_000.0) * float(pricing["input"])
        output_cost = (out_tok_i / 1_000_000.0) * float(pricing["output"])
        total_cost = input_cost + output_cost
        return {
            "model_key": key,
            "input_tokens": in_tok_i,
            "cached_input_tokens": cached_i,
            "uncached_input_tokens": uncached_i,
            "output_tokens": out_tok_i,
            "reasoning_output_tokens": reasoning_i,
            "visible_output_tokens": visible_i,
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
        }

    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[str]] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Tuple[Any, int, int]:
        user_text = str(user_msg or "")
        if schema is not None:
            user_text += self._json_only_suffix()

        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in (images or []):
            content.append(await self._image_block_from_src(img))

        base_req: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": int(self.max_tokens),
            "messages": [{"role": "user", "content": content}],
        }
        if system_msg:
            base_req["system"] = system_msg
        if self.temperature is not None:
            base_req["temperature"] = float(self.temperature)

        last_err: Optional[BaseException] = None
        resp = None
        attempts = self._thinking_attempts(base_req)
        for idx, (label, req) in enumerate(attempts):
            try:
                resp = await self._messages_create_with_retries(req)
                if idx > 0:
                    print(
                        f"WARNING: Claude request recovered via fallback mode={label} model={self.model_name}."
                    )
                break
            except Exception as exc:
                last_err = exc
                if idx < len(attempts) - 1:
                    print(
                        f"WARNING: Claude request mode={label} failed for model={self.model_name}; "
                        f"trying next fallback. error={type(exc).__name__}: {exc}"
                    )
                    continue
                raise

        if resp is None:
            raise RuntimeError(f"Claude request failed: {last_err}")

        usage = getattr(resp, "usage", None)
        in_tok, out_tok, cached_in_tok, reasoning_out_tok = self._extract_usage_tokens(usage)

        txt = _extract_text_from_message(resp)
        try:
            obj: Any = json.loads(txt) if txt else {}
        except Exception:
            obj = txt or {}

        if schema is not None:
            try:
                validated = schema.model_validate(obj)
                obj = validated.model_dump()
            except Exception:
                pass

        cost_info = self._estimate_cost(
            int(in_tok or 0),
            int(out_tok or 0),
            int(cached_in_tok or 0),
            int(reasoning_out_tok or 0),
        )
        async with self._stats_lock:
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

        return obj, int(in_tok or 0), int(out_tok or 0)

    async def generate_async(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[Any, int, int]]:
        print_cost: bool = bool(kwargs.pop("print_cost", self.print_cost))
        max_concurrency = int(kwargs.pop("max_concurrency", self.max_concurrency))
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer.")
        sem = asyncio.Semaphore(max_concurrency)

        async with self._stats_lock:
            base_cached = int(self.cached_input_tokens_used)
            base_reasoning = int(self.reasoning_output_tokens_used)
            base_cost = float(self.total_cost_usd)
            base_costed = int(self.costed_requests)
            base_uncosted = int(self.uncosted_requests)

        async def _wrap(idx: int, sys_msg: str, user_msg: str, img_list: Optional[List[str]]):
            async with sem:
                try:
                    out = await self._generate_single(sys_msg, user_msg, img_list, schema)
                    return idx, out
                except Exception as exc:
                    placeholder = ({"__error__": f"{type(exc).__name__}: {exc}"}, 0, 0)
                    return idx, placeholder

        tasks = []
        for i, tup in enumerate(prompts):
            sys_msg, user_msg = tup[0], tup[1]
            img_list = tup[2] if len(tup) > 2 else None
            tasks.append(asyncio.create_task(_wrap(i, sys_msg, user_msg, img_list)))

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
                1
                for r in final_results
                if isinstance(r, (list, tuple))
                and len(r) > 0
                and isinstance(r[0], dict)
                and "__error__" in r[0]
            )

            async with self._stats_lock:
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
                summary += f" | priced={call_costed} unpriced={unpriced_count}"
            if call_failed > 0:
                summary += f" | failed={call_failed}"
            print(summary)

        return final_results

    def generate(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[Any, int, int]]:
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
