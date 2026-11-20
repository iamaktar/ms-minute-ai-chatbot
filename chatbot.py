"""Core LLM client wrapper, streaming logic, image generation, and error handling."""

from __future__ import annotations

import time
from collections.abc import Iterator

from google import genai
from google.genai import errors, types

from config import Settings
from memory import ConversationMemory

FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
)

IMAGE_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image",
    "gemini-2.5-flash-image",
)


class ChatbotError(RuntimeError):
    """Base exception for user-facing chatbot failures."""


class RateLimitError(ChatbotError):
    """Raised when the Gemini API rate limit or quota is exceeded."""


class NetworkTimeoutError(ChatbotError):
    """Raised when an API request times out or encounters a network failure."""


class GeminiChatbot:
    """High-level wrapper around the Google GenAI SDK with streaming, memory, and images."""

    def __init__(
        self,
        settings: Settings,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._settings = settings
        self._memory = memory or ConversationMemory(
            max_messages=settings.max_history_messages
        )
        self._system_instruction: str = settings.system_instruction
        self._client = genai.Client(
            api_key=settings.api_key,
            http_options=types.HttpOptions(
                timeout=settings.request_timeout_seconds * 1000
            ),
        )

    @property
    def model_name(self) -> str:
        """Return the active Gemini model name."""
        return self._settings.model_name

    @property
    def system_instruction(self) -> str:
        """Return the active system instruction."""
        return self._system_instruction

    @property
    def memory(self) -> ConversationMemory:
        """Return the underlying ConversationMemory instance."""
        return self._memory

    def set_model_name(self, model_name: str) -> None:
        """Switch the active Gemini model for subsequent turns."""
        cleaned = model_name.strip()
        if not cleaned:
            raise ValueError("Model name cannot be empty.")
        self._settings.model_name = cleaned

    def set_system_instruction(self, instruction: str) -> None:
        """Update the active system persona for subsequent turns."""
        cleaned = instruction.strip()
        if not cleaned:
            raise ValueError("System instruction cannot be empty.")
        self._system_instruction = cleaned

    def clear_history(self) -> None:
        """Reset the conversation memory."""
        self._memory.clear()

    def _candidate_models(self) -> list[str]:
        """Return the primary model followed by unique fallback models."""
        models = [self._settings.model_name]
        for fallback in FALLBACK_MODELS:
            if fallback not in models:
                models.append(fallback)
        return models

    def generate_image(self, prompt: str) -> bytes:
        """Generate an image from a text prompt using Gemini native image models.

        Args:
            prompt: Text description of the image to generate.

        Returns:
            Raw image bytes (PNG/JPEG).

        Raises:
            ValueError: If prompt is empty.
            RateLimitError: If all image models hit rate limits.
            ChatbotError: If image generation fails or returns no image data.
        """
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise ValueError("Image prompt cannot be empty.")

        last_api_error: errors.APIError | None = None

        for model in IMAGE_MODELS:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=cleaned_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE"],
                    ),
                )

                parts = getattr(response, "parts", None)
                if not parts and getattr(response, "candidates", None):
                    parts = response.candidates[0].content.parts

                for part in parts or []:
                    inline_data = getattr(part, "inline_data", None)
                    if inline_data and getattr(inline_data, "data", None):
                        return bytes(inline_data.data)

            except errors.APIError as exc:
                status_code = getattr(exc, "code", None)
                if status_code in (404, 429):
                    last_api_error = exc
                    continue
                raise ChatbotError(
                    f"Image API error ({status_code or 'unknown'}): {exc.message}"
                ) from exc
            except (TimeoutError, OSError) as exc:
                raise NetworkTimeoutError(
                    f"Network error during image generation: {exc}"
                ) from exc

        if last_api_error and getattr(last_api_error, "code", None) == 429:
            raise RateLimitError(
                f"Image generation rate limit exceeded. ({last_api_error.message})"
            ) from last_api_error

        detail = (
            f" ({last_api_error.message})"
            if last_api_error and getattr(last_api_error, "message", None)
            else ""
        )
        raise ChatbotError(f"Could not generate an image.{detail}")

    def stream_reply(self, user_input: str) -> Iterator[str]:
        """Send a user message and yield response text chunks as they arrive."""
        cleaned_input = user_input.strip()
        if not cleaned_input:
            raise ValueError("Message cannot be empty.")

        self._memory.add_user_message(cleaned_input)
        config = types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            temperature=self._settings.temperature,
        )
        contents = self._memory.to_genai_contents()

        collected_chunks: list[str] = []
        last_api_error: errors.APIError | None = None
        retry_delays = (2.0, 5.0)

        try:
            for model in self._candidate_models():
                for attempt in range(len(retry_delays) + 1):
                    try:
                        response_stream = self._client.models.generate_content_stream(
                            model=model,
                            contents=contents,
                            config=config,
                        )
                        for chunk in response_stream:
                            text = getattr(chunk, "text", None)
                            if text:
                                collected_chunks.append(text)
                                yield text

                        full_response = "".join(collected_chunks).strip()
                        if full_response:
                            self._memory.add_model_message(full_response)
                        else:
                            self._memory.pop_last_message()
                        return

                    except errors.APIError as exc:
                        status_code = getattr(exc, "code", None)
                        if status_code == 429 and not collected_chunks:
                            last_api_error = exc
                            if attempt < len(retry_delays):
                                time.sleep(retry_delays[attempt])
                                continue
                            break
                        if status_code == 404 and not collected_chunks:
                            last_api_error = exc
                            break
                        if status_code in (408, 504):
                            self._rollback_user_message(collected_chunks)
                            raise NetworkTimeoutError(
                                "The request timed out. Please check your connection and retry."
                            ) from exc
                        self._rollback_user_message(collected_chunks)
                        raise ChatbotError(
                            f"Gemini API error ({status_code or 'unknown'}): {exc.message}"
                        ) from exc

            self._rollback_user_message(collected_chunks)
            status_code = getattr(last_api_error, "code", None) if last_api_error else None
            detail = (
                f" ({last_api_error.message})"
                if last_api_error and getattr(last_api_error, "message", None)
                else ""
            )
            if status_code == 429:
                raise RateLimitError(
                    f"Rate limit or quota exceeded on all models. Wait 1 minute or check your Google AI Studio quota.{detail}"
                ) from last_api_error
            raise ChatbotError(
                f"No available Gemini model succeeded.{detail}"
            ) from last_api_error

        except TimeoutError as exc:
            self._rollback_user_message(collected_chunks)
            raise NetworkTimeoutError(
                "Network timeout while waiting for Gemini. Please try again."
            ) from exc

        except OSError as exc:
            self._rollback_user_message(collected_chunks)
            raise NetworkTimeoutError(f"Network error: {exc}") from exc

        except KeyboardInterrupt:
            if collected_chunks:
                self._memory.add_model_message("".join(collected_chunks))
            else:
                self._memory.pop_last_message()
            raise

    def _rollback_user_message(self, collected_chunks: list[str]) -> None:
        """Roll back the pending user turn if no model output was generated."""
        if not collected_chunks:
            self._memory.pop_last_message()