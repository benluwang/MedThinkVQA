from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import pathlib
import random
from typing import Any, List, Optional, Tuple, Type

import nest_asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from pydantic import BaseModel
from tqdm import tqdm

try:
    from model.model import Model
except ImportError:
    # Backward compatibility: current base class is named `LLM` in this repo.
    from model.model import LLM as Model

nest_asyncio.apply()


def _get_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Always use the Gemini Developer API (NOT Vertex AI).

    This explicitly sets `vertexai=False` and reads `GEMINI_API_KEY`.
    """
    try:
        load_dotenv()
    except AssertionError:
        # python-dotenv can raise on stdin/eval contexts (e.g. `python - <<'PY'`).
        load_dotenv(".env")
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing `GEMINI_API_KEY` (set it in `.env` or the environment).")
    return genai.Client(vertexai=False, api_key=api_key)


def _extract_text(resp: types.GenerateContentResponse) -> str:
    """
    Extract only text parts from a GenerateContent response.

    We avoid using `resp.text` because the SDK logs a warning when the response
    contains non-text parts (e.g. `thought_signature`).
    """
    for cand in getattr(resp, "candidates", []) or []:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        texts: list[str] = []
        for part in parts:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                texts.append(t)
        if texts:
            return "".join(texts)
    return ""


def _grounding_tool_for_model(model_name: str) -> types.Tool:
    """
    Build a Google Search grounding tool.

    Note: Some preview/experimental models do not support Google Search grounding.
    """
    if model_name.startswith("gemini-1.5"):
        return types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
    return types.Tool(google_search=types.GoogleSearch())


def _guess_mime_type(src: str) -> str:
    base = str(src).split("?", 1)[0].split("#", 1)[0]
    mime = mimetypes.guess_type(base)[0]
    return mime or "application/octet-stream"


def _image_part_from_source(src: str) -> types.Part:
    if not isinstance(src, str) or not src.strip():
        raise ValueError("Image source must be a non-empty string.")
    src = src.strip()

    if src.startswith(("http://", "https://", "gs://", "file://")):
        return types.Part.from_uri(file_uri=src, mime_type=_guess_mime_type(src))

    if src.startswith("data:"):
        header, sep, payload = src.partition(",")
        if not sep:
            raise ValueError("Invalid data URL for image input.")
        mime = header[5:].split(";", 1)[0] or "application/octet-stream"
        if ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported for image input.")
        try:
            raw = base64.b64decode(payload, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 image data URL: {e}") from e
        return types.Part.from_bytes(data=raw, mime_type=mime)

    path = pathlib.Path(src).expanduser()
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    mime = _guess_mime_type(str(path))
    raw = path.read_bytes()
    return types.Part.from_bytes(data=raw, mime_type=mime)


def _build_contents(user_msg: str, images: Optional[List[str]]) -> Any:
    """
    Build Gemini contents payload:
    - text-only -> plain user text
    - text+images -> UserContent(parts=[text, image, ...])
    """
    if not images:
        return user_msg

    parts: list[types.Part] = []
    if user_msg:
        parts.append(types.Part.from_text(text=user_msg))
    for img in images:
        parts.append(_image_part_from_source(img))

    if not parts:
        parts.append(types.Part.from_text(text=""))
    return types.UserContent(parts=parts)


class GeminiModel(Model):
    DEFAULT_INVALID_RESPONSE_RETRIES = 2

    # Gemini Developer API Standard pricing (USD per 1M tokens), checked 2026-03-01:
    # https://ai.google.dev/gemini-api/docs/pricing
    MODEL_PRICING_USD_PER_1M = {
        "gemini-3-flash-preview": {
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "tiers": [
                # Standard, text/image/video pricing.
                {"max_input_tokens": None, "input": 0.50, "cached_input": 0.05, "output": 3.0},
            ],
        },
        "gemini-2.5-pro": {
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "tiers": [
                {"max_input_tokens": 200_000, "input": 1.25, "cached_input": 0.125, "output": 10.0},
                {"max_input_tokens": None, "input": 2.50, "cached_input": 0.25, "output": 15.0},
            ],
        },
        "gemini-2.5-flash": {
            "source": "https://ai.google.dev/gemini-api/docs/pricing",
            "tiers": [
                # Standard, text/image/video pricing.
                {"max_input_tokens": None, "input": 0.30, "cached_input": 0.03, "output": 2.5},
            ],
        },
    }

    def __init__(
        self,
        model_name: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        google_search: bool = False,
        thinking_budget: Optional[int] = None,
        max_retries: int = 5,
        max_invalid_response_retries: int = DEFAULT_INVALID_RESPONSE_RETRIES,
        print_cost: bool = True,
    ) -> None:
        base_model_name = str(model_name or "").strip()
        if "/" not in base_model_name:
            base_model_name = f"Google/{base_model_name}"
        super().__init__(model_name=base_model_name, temperature=temperature)
        self.client = _get_client()
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens is not None else None
        self.google_search = bool(google_search)
        self.thinking_budget = int(thinking_budget) if thinking_budget is not None else None
        self.max_retries = int(max_retries)
        self.max_invalid_response_retries = max(0, int(max_invalid_response_retries))
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
            f"{self.model_name} | temperature={self.temperature} max_output_tokens={self.max_output_tokens} "
            f"google_search={self.google_search} thinking_budget={self.thinking_budget}"
        )

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

    def _pricing_key(self) -> Optional[str]:
        name = str(self.model_name or "").strip().lower()
        if name.startswith("gemini-3-flash-preview"):
            return "gemini-3-flash-preview"
        if name.startswith("gemini-2.5-pro"):
            return "gemini-2.5-pro"
        if name.startswith("gemini-2.5-flash"):
            return "gemini-2.5-flash"
        return name if name in self.MODEL_PRICING_USD_PER_1M else None

    @staticmethod
    def _extract_finish_reasons(resp: Any) -> List[str]:
        out: List[str] = []
        for cand in getattr(resp, "candidates", []) or []:
            reason = getattr(cand, "finish_reason", None)
            if reason is None:
                continue
            s = str(reason).strip()
            if s:
                out.append(s)
        return out

    @staticmethod
    def _is_empty_payload(obj: Any, txt: str) -> bool:
        if isinstance(txt, str) and txt.strip():
            return False
        if obj is None:
            return True
        if isinstance(obj, str):
            return not obj.strip()
        if isinstance(obj, (dict, list, tuple, set)):
            return len(obj) == 0
        return False

    def _pricing_tier(self, input_tokens: int) -> Optional[dict]:
        key = self._pricing_key()
        if key is None:
            return None
        entry = self.MODEL_PRICING_USD_PER_1M.get(key) or {}
        tiers = entry.get("tiers", [])
        if not isinstance(tiers, list):
            return None
        for tier in tiers:
            if not isinstance(tier, dict):
                continue
            max_tok = tier.get("max_input_tokens")
            if max_tok is None or int(input_tokens) <= int(max_tok):
                out = dict(tier)
                out["model_key"] = key
                out["source"] = entry.get("source")
                return out
        return None

    def _extract_usage_tokens(self, usage: Any) -> Tuple[int, int, int, int, int]:
        """
        Extract:
          input_tokens (prompt + tool_use_prompt),
          output_tokens (candidates + thoughts),
          cached_input_tokens,
          reasoning_output_tokens (thoughts),
          visible_output_tokens (candidates)
        from Gemini usage_metadata.
        """
        prompt_tok = self._as_int(getattr(usage, "prompt_token_count", 0))
        tool_use_prompt_tok = self._as_int(getattr(usage, "tool_use_prompt_token_count", 0))
        input_tok = max(0, prompt_tok + tool_use_prompt_tok)

        cached_in_tok = self._as_int(getattr(usage, "cached_content_token_count", 0))
        cached_in_tok = max(0, min(cached_in_tok, input_tok))

        thoughts_tok = self._as_int(getattr(usage, "thoughts_token_count", 0))

        cand_raw = getattr(usage, "candidates_token_count", None)
        if cand_raw is None:
            # Backward compatibility with older SDK naming.
            cand_raw = getattr(usage, "response_token_count", None)

        if cand_raw is None:
            total_tok = self._as_int(getattr(usage, "total_token_count", 0))
            visible_out_tok = max(0, total_tok - input_tok - thoughts_tok)
        else:
            visible_out_tok = max(0, self._as_int(cand_raw))

        output_tok = max(0, visible_out_tok + max(0, thoughts_tok))
        reasoning_out_tok = max(0, min(thoughts_tok, output_tok))
        return input_tok, output_tok, cached_in_tok, reasoning_out_tok, visible_out_tok

    def _estimate_cost(
        self, in_tok: int, out_tok: int, cached_in_tok: int, reasoning_out_tok: int = 0
    ) -> Optional[dict]:
        """
        Estimate request cost from Gemini token usage and static pricing table.
        Returns None when pricing is unavailable for the current model.
        """
        tier = self._pricing_tier(in_tok)
        if tier is None:
            return None

        in_tok_i = max(0, self._as_int(in_tok))
        out_tok_i = max(0, self._as_int(out_tok))
        cached_i = min(max(0, self._as_int(cached_in_tok)), in_tok_i)
        uncached_i = max(0, in_tok_i - cached_i)
        reasoning_i = min(max(0, self._as_int(reasoning_out_tok)), out_tok_i)
        visible_i = max(0, out_tok_i - reasoning_i)

        input_rate = tier.get("input")
        cached_rate_raw = tier.get("cached_input")
        output_rate = tier.get("output")

        input_cost = self._cost_component(uncached_i, input_rate)
        if cached_i <= 0:
            cached_cost = 0.0
        elif cached_rate_raw is None:
            cached_cost = None
        else:
            cached_cost = self._cost_component(cached_i, cached_rate_raw)
        output_cost = self._cost_component(out_tok_i, output_rate)

        if input_cost is None or cached_cost is None or output_cost is None:
            total_cost = None
        else:
            total_cost = input_cost + cached_cost + output_cost

        return {
            "model_key": tier.get("model_key"),
            "pricing_source": tier.get("source"),
            "pricing_input_rate_per_1m": input_rate,
            "pricing_cached_input_rate_per_1m": cached_rate_raw,
            "pricing_output_rate_per_1m": output_rate,
            "pricing_tier_max_input_tokens": tier.get("max_input_tokens"),
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

    async def _generate_with_retries(
        self, *, model: str, contents: Any, config: Optional[types.GenerateContentConfig]
    ) -> types.GenerateContentResponse:
        last_err: Optional[BaseException] = None
        for attempt in range(max(self.max_retries, 1)):
            try:
                return await self.client.aio.models.generate_content(model=model, contents=contents, config=config)
            except (ClientError, Exception) as e:
                last_err = e
                if attempt >= self.max_retries - 1:
                    raise
                backoff = min(2**attempt, 10) + random.random()
                await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _generate_single(
        self,
        system_msg: str,
        user_msg: str,
        images: Optional[List[str]] = None,
        schema: Optional[Type[BaseModel]] = None,
    ) -> Tuple[Any, int, int]:
        cfg_kwargs: dict[str, Any] = {}
        if system_msg:
            cfg_kwargs["system_instruction"] = system_msg
        if self.temperature is not None:
            cfg_kwargs["temperature"] = float(self.temperature)
        if self.max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(self.max_output_tokens)
        if self.thinking_budget is not None:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
        if self.google_search:
            cfg_kwargs["tools"] = [_grounding_tool_for_model(self.model_name)]

        if schema is not None:
            cfg_kwargs["response_mime_type"] = "application/json"
            cfg_kwargs["response_schema"] = schema

        config = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None
        contents = _build_contents(user_msg, images)
        total_in_tok = 0
        total_out_tok = 0
        total_cached_in_tok = 0
        total_reasoning_out_tok = 0
        total_cost_usd = 0.0
        has_priced_attempt = False
        has_unpriced_attempt = False
        attempts_made = 0

        attempts = max(1, int(self.max_invalid_response_retries) + 1)
        obj: Any = {"__error__": "EmptyResponseError: uninitialized Gemini result."}
        last_schema_err: Optional[BaseException] = None
        last_txt = ""

        for attempt_idx in range(attempts):
            resp = await self._generate_with_retries(model=self.model_name, contents=contents, config=config)
            attempts_made += 1

            usage = getattr(resp, "usage_metadata", None)
            in_tok, out_tok, cached_in_tok, reasoning_out_tok, _visible_out_tok = self._extract_usage_tokens(usage)
            in_tok = int(in_tok or 0)
            out_tok = int(out_tok or 0)
            cached_in_tok = int(cached_in_tok or 0)
            reasoning_out_tok = int(reasoning_out_tok or 0)

            total_in_tok += in_tok
            total_out_tok += out_tok
            total_cached_in_tok += cached_in_tok
            total_reasoning_out_tok += reasoning_out_tok

            attempt_cost = self._estimate_cost(in_tok, out_tok, cached_in_tok, reasoning_out_tok)
            if attempt_cost is not None and attempt_cost.get("total_cost_usd") is not None:
                total_cost_usd += float(attempt_cost["total_cost_usd"])
                has_priced_attempt = True
            else:
                has_unpriced_attempt = True

            parsed = getattr(resp, "parsed", None)
            txt = _extract_text(resp)
            last_txt = txt
            if parsed is None:
                try:
                    obj = json.loads(txt) if txt else {}
                except Exception:
                    obj = txt or {}
            else:
                if isinstance(parsed, BaseModel):
                    obj = parsed.model_dump()
                elif hasattr(parsed, "model_dump"):
                    obj = parsed.model_dump()
                else:
                    obj = parsed

            payload_empty = self._is_empty_payload(obj, txt)
            schema_ok = True
            if schema is not None and not (isinstance(obj, dict) and "__error__" in obj):
                try:
                    validated = schema.model_validate(obj)
                    obj = validated.model_dump()
                except Exception as e:
                    schema_ok = False
                    last_schema_err = e

            if not payload_empty and schema_ok:
                break

            if attempt_idx < attempts - 1:
                finish_reasons = self._extract_finish_reasons(resp)
                finish_info = f" finish={finish_reasons}" if finish_reasons else ""
                reason_bits = []
                if payload_empty:
                    reason_bits.append("empty_payload")
                if not schema_ok:
                    reason_bits.append("schema_invalid")
                reason = ",".join(reason_bits) or "invalid_output"
                print(
                    f"WARNING: Gemini returned {reason} for model={self.model_name}; "
                    f"retrying ({attempt_idx + 1}/{attempts - 1}).{finish_info}"
                )
                backoff = min(2**attempt_idx, 4) + random.random()
                await asyncio.sleep(backoff)
                continue

            if payload_empty:
                finish_reasons = self._extract_finish_reasons(resp)
                finish_info = f" finish={finish_reasons}" if finish_reasons else ""
                obj = {
                    "__error__": (
                        f"EmptyResponseError: Gemini returned no parseable content "
                        f"after {attempts} attempt(s).{finish_info}"
                    )
                }
            elif schema is not None and not schema_ok:
                if isinstance(obj, str) and obj.strip():
                    pass
                elif isinstance(last_txt, str) and last_txt.strip():
                    obj = last_txt.strip()
                else:
                    err_msg = f"{type(last_schema_err).__name__}: {last_schema_err}" if last_schema_err else "unknown"
                    obj = {"__error__": f"SchemaValidationError: {err_msg}"}
            break

        async with self._stats_lock:
            total = int(total_in_tok or 0) + int(total_out_tok or 0)
            self.tokens_used += total
            self.cached_input_tokens_used += int(total_cached_in_tok or 0)
            self.reasoning_output_tokens_used += int(total_reasoning_out_tok or 0)
            self.requests_made += max(1, int(attempts_made))
            if has_priced_attempt:
                self.total_cost_usd += float(total_cost_usd)
                self.costed_requests += 1
            if has_unpriced_attempt or not has_priced_attempt:
                self.uncosted_requests += 1

        return obj, int(total_in_tok or 0), int(total_out_tok or 0)

    async def generate_async(
        self,
        prompts: List[Tuple[str, str, Optional[List[str]]]],
        schema: Optional[Type[BaseModel]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[Tuple[Any, int, int]]:
        print_cost: bool = bool(kwargs.pop("print_cost", self.print_cost))
        max_concurrency: int = int(kwargs.pop("max_concurrency", 16))
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
                except Exception as e:
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
            completed = 0
            for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"{self.model_name} generating..."):
                idx, res = await fut
                results[idx] = res
                completed += 1
                print(f"[gemini] completed {completed}/{len(prompts)}", flush=True)
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
            if self.google_search:
                summary += " | note=search-grounding fees are not included"
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
