#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

"""Completion of prompts (using large language models)."""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast
from pathlib import Path

import openai
from diskcache import Cache
from openai import OpenAIError
from pydantic import BaseModel

RATE_LIMIT_RETRIES = 5
DEFAULT_CACHE_DIR = Path("/Users/joestacey/.cache/lucid")


class Prompt(BaseModel):
    # The prefix to be completed.
    prefix: str

    # Start text to add to the prefix, this will also be pre-pended to the completion.
    start_text: str = ""

    # Text to stop completion on.
    stop_texts: Optional[List[str]] = None


class CompletionError(ValueError):
    pass


class CompletionApiError(CompletionError):
    pass


class CompletionTooShortError(CompletionError):
    pass


class CompletionGetPromptError(CompletionError):
    pass


class Completer(ABC):
    """Class that can be used to complete prompts."""

    _cache: CompletionCache

    async def complete(self, prompt: Prompt, use_cache: bool = True, max_retries: int = 1) -> str:
        """Complete the prompt, using the cache."""
        return await self._cache.cached_complete(self._complete, prompt, use_cache, max_retries)

    @abstractmethod
    async def _complete(self, prompt: Prompt) -> str:
        """Implementation of prompt completion, used by self.complete."""

    async def get_prompt(self) -> Prompt:
        """Returns a prompt to complete."""
        raise CompletionGetPromptError("get_prompt not implemented for this completer")

    def set_completion_response(self, response: str) -> None:
        """Sets the completion response."""
        pass


def _make_cache_key(prompt: Prompt) -> str:
    return prompt.json()


class CompletionCache:
    """A cache for completion results, that can be used by Completer implementations.
    Also implements retrying.
    """

    def __init__(self, cache_dir: Optional[Path]):
        self._cache = self._make_cache(cache_dir)

    @staticmethod
    def _make_cache(cache_dir: Optional[Path]) -> Optional[Cache]:
        if cache_dir is None:
            return None
        cache_dir.mkdir(exist_ok=True, parents=True)
        assert cache_dir.is_dir()
        return Cache(str(cache_dir))

    async def cached_complete(
        self,
        complete_fn: Callable[[Prompt], Awaitable[str]],
        prompt: Prompt,
        use_cache: bool = True,
        max_retries: int = 1,
    ) -> str:
        """Use the given complete_fn, or return a cached completion."""
        key = _make_cache_key(prompt)
        if self._cache is not None and use_cache:
            if key in self._cache:
                return cast(str, self._cache[key])

        completion: Optional[str] = None
        for i in range(max_retries):
            try:
                completion = await complete_fn(prompt)
                break
            except CompletionError as e:
                if i == max_retries - 1:
                    raise e

        assert completion is not None
        if self._cache is not None:
            self._cache[key] = completion

        return completion


class OpenAiChatCompleter(Completer):
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 100,
        best_of_n: int = 1,
        temperature: float = 0.7,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR / "openai",
    ):
        self._model_name = model_name
        self._max_tokens = max_tokens
        self._best_of_n = best_of_n
        self._temperature = temperature
        self._cache = CompletionCache(self._full_cache_dir(cache_dir))
        if "OPENAI_API_KEY" not in os.environ:
            raise RuntimeError(f"Set OPENAI_API_KEY env variable to use {self.__class__.__name__}")

    def _full_cache_dir(self, cache_dir: Optional[Path]) -> Optional[Path]:
        if cache_dir is None:
            return None
        return (
            cache_dir
            / f"chat__{self._model_name}"
            / f"mt_{self._max_tokens}__n_{self._best_of_n}__temp_{self._temperature:.3f}"
        )

    async def _complete(self, prompt: Prompt) -> str:
        """Implementation that is wrapped by `complete`, potentially cached."""
        if prompt.start_text:
            raise ValueError(f"Start text is not implemented for OpenAiChatCompleter.")

        response: Optional[Dict[str, Any]] = None

        retries = 10

        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self._model_name,
                    messages=[{"role": "system", "content": prompt.prefix}],
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    stop=prompt.stop_texts,
                    n=self._best_of_n,
                )
            except OpenAIError as e:
                if "Rate limit" in str(e) and attempt != RATE_LIMIT_RETRIES:
                    print(
                        f"\nOpenAI Rate limit reached. Waiting for 1 minute before retrying. (number of retries remaining: {RATE_LIMIT_RETRIES-(attempt+1)})\n"
                    )
                    time.sleep(60)
                else:
                    raise CompletionApiError(f"OpenAIError: {repr(e)}")
            else:
                break

        if not len(response.get("choices", [])) >= 1:
            raise CompletionApiError("No completion returned from API")

        top_completion = response["choices"][0]
        message = top_completion["message"]
        if message["role"] != "assistant":
            raise CompletionApiError(
                f"API returned message with role '{message['role']}', expected to be 'assistant'."
            )

        text = cast(str, message["content"])

        ends_with_stop_text = prompt.stop_texts and any(
            text.endswith(stop_text) for stop_text in prompt.stop_texts
        )
        if top_completion.get("finish_reason") == "length" and not ends_with_stop_text:
            raise CompletionTooShortError("API reached token limit before returning answer")

        return text
