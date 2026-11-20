"""Voice engine using Edge Neural TTS, Pygame audio playback, and SpeechRecognition."""

from __future__ import annotations

import asyncio
import os
import queue
import re
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import edge_tts
import pygame
import speech_recognition as sr


class VoiceError(RuntimeError):
    """Raised when microphone capture or speech recognition fails."""


@dataclass(frozen=True, slots=True)
class VoiceOption:
    """Represents a selectable TTS voice and matching STT language code."""

    label: str
    tts_voice: str
    stt_language: str
    sample_text: str


VOICE_OPTIONS: dict[str, VoiceOption] = {
    "English (India) - Neerja": VoiceOption(
        label="English (India) - Neerja",
        tts_voice="en-IN-NeerjaNeural",
        stt_language="en-IN",
        sample_text="Hello! I am Ms. Minute, and my voice is working.",
    ),
    "English (US) - Aria": VoiceOption(
        label="English (US) - Aria",
        tts_voice="en-US-AriaNeural",
        stt_language="en-US",
        sample_text="Hello! I am Ms. Minute, ready to help you.",
    ),
    "English (UK) - Sonia": VoiceOption(
        label="English (UK) - Sonia",
        tts_voice="en-GB-SoniaNeural",
        stt_language="en-GB",
        sample_text="Hello! I am Ms. Minute, at your service.",
    ),
    "Bengali (India) - Tanishaa": VoiceOption(
        label="Bengali (India) - Tanishaa",
        tts_voice="bn-IN-TanishaaNeural",
        stt_language="bn-IN",
        sample_text="নমস্কার! আমি মিস মিনিট, এখন আমি বাংলায় কথা বলতে পারি।",
    ),
    "Hindi (India) - Swara": VoiceOption(
        label="Hindi (India) - Swara",
        tts_voice="hi-IN-SwaraNeural",
        stt_language="hi-IN",
        sample_text="नमस्ते! मैं मिस मिनट हूँ, और अब मैं हिंदी में बात कर सकती हूँ।",
    ),
}

DEFAULT_VOICE_LABEL = "English (India) - Neerja"

BENGALI_VOICE = "bn-IN-TanishaaNeural"
HINDI_VOICE = "hi-IN-SwaraNeural"
ENGLISH_FALLBACK_VOICE = "en-IN-NeerjaNeural"

_BENGALI_PATTERN = re.compile(r"[\u0980-\u09FF]")
_HINDI_PATTERN = re.compile(r"[\u0900-\u097F]")


def clean_text_for_speech(text: str) -> str:
    """Strip Markdown symbols and code blocks so spoken replies sound natural."""
    cleaned = re.sub(r"```[\s\S]*?```", " code block omitted ", text)
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[#*_~>|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def detect_tts_voice(text: str, preferred_voice: str) -> str:
    """Detect Bengali, Hindi, or English script and return the matching neural voice."""
    bengali_count = len(_BENGALI_PATTERN.findall(text))
    hindi_count = len(_HINDI_PATTERN.findall(text))

    if bengali_count > 0 and bengali_count >= hindi_count:
        return BENGALI_VOICE

    if hindi_count > 0 and hindi_count > bengali_count:
        return HINDI_VOICE

    if preferred_voice.startswith("en-"):
        return preferred_voice

    return ENGLISH_FALLBACK_VOICE


class VoiceEngine:
    """Thread-safe neural speech synthesis and microphone recognition manager."""

    def __init__(
        self,
        voice_label: str = DEFAULT_VOICE_LABEL,
        on_speaking_change: Callable[[bool], None] | None = None,
    ) -> None:
        self._voice: VoiceOption = VOICE_OPTIONS.get(
            voice_label, VOICE_OPTIONS[DEFAULT_VOICE_LABEL]
        )
        self._on_speaking_change = on_speaking_change
        self._recognizer = sr.Recognizer()
        self._recognizer.dynamic_energy_threshold = True
        self._recognizer.pause_threshold = 0.8

        if not pygame.mixer.get_init():
            pygame.mixer.init()

        self._tts_queue: queue.Queue[tuple[str, str] | None] = queue.Queue()
        self._stop_event = threading.Event()
        self._cancel_current = threading.Event()

        self._tts_thread = threading.Thread(
            target=self._tts_worker,
            name="MsMinute-NeuralTTS",
            daemon=True,
        )
        self._tts_thread.start()

    @property
    def current_voice(self) -> VoiceOption:
        """Return the currently selected VoiceOption."""
        return self._voice

    def set_voice(self, voice_label: str) -> VoiceOption:
        """Switch the active TTS voice and STT language."""
        if voice_label not in VOICE_OPTIONS:
            raise ValueError(f"Unknown voice option: {voice_label}")
        self._voice = VOICE_OPTIONS[voice_label]
        return self._voice

    def _notify_speaking(self, is_speaking: bool) -> None:
        if self._on_speaking_change is not None:
            try:
                self._on_speaking_change(is_speaking)
            except Exception:  # noqa: BLE001
                pass

    def _tts_worker(self) -> None:
        """Background worker that synthesizes and plays queued speech."""
        while not self._stop_event.is_set():
            item = self._tts_queue.get()
            if item is None:
                self._tts_queue.task_done()
                break

            text, tts_voice = item
            self._cancel_current.clear()
            temp_path: str | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmp_file:
                    temp_path = tmp_file.name

                asyncio.run(self._synthesize_to_file(text, tts_voice, temp_path))

                if self._cancel_current.is_set() or self._stop_event.is_set():
                    continue

                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                self._notify_speaking(True)

                while pygame.mixer.music.get_busy():
                    if self._cancel_current.is_set() or self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)

                pygame.mixer.music.unload()
            except Exception:  # noqa: BLE001
                try:
                    pygame.mixer.music.stop()
                    pygame.mixer.music.unload()
                except Exception:  # noqa: BLE001
                    pass
            finally:
                self._notify_speaking(False)
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                self._tts_queue.task_done()

    @staticmethod
    async def _synthesize_to_file(text: str, tts_voice: str, output_path: str) -> None:
        """Generate an MP3 audio file using Microsoft Edge Neural TTS."""
        communicate = edge_tts.Communicate(text=text, voice=tts_voice)
        await communicate.save(output_path)

    def speak(self, text: str, auto_detect: bool = True) -> None:
        """Queue a message to be spoken aloud asynchronously."""
        spoken_text = clean_text_for_speech(text)
        if not spoken_text:
            return

        selected_voice = (
            detect_tts_voice(spoken_text, self._voice.tts_voice)
            if auto_detect
            else self._voice.tts_voice
        )
        self._tts_queue.put((spoken_text, selected_voice))

    def speak_sample(self) -> None:
        """Speak the sample phrase for the currently selected voice."""
        self.stop_speaking()
        self.speak(self._voice.sample_text, auto_detect=False)

    def stop_speaking(self) -> None:
        """Clear pending speech and immediately stop active playback."""
        self._cancel_current.set()
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break

        try:
            if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
        except Exception:  # noqa: BLE001
            pass
        self._notify_speaking(False)

    def listen(self, timeout: int = 6, phrase_time_limit: int = 15) -> str:
        """Capture audio from the microphone and transcribe it in the active language."""
        try:
            with sr.Microphone() as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.4)
                audio = self._recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit,
                )
        except sr.WaitTimeoutError as exc:
            raise VoiceError("No speech detected. Click Speak and try again.") from exc
        except (OSError, AttributeError) as exc:
            raise VoiceError(
                f"Microphone unavailable. Check your mic and PyAudio setup: {exc}"
            ) from exc

        try:
            return self._recognizer.recognize_google(
                audio,
                language=self._voice.stt_language,
            ).strip()
        except sr.UnknownValueError as exc:
            raise VoiceError("Could not understand the audio. Please try again.") from exc
        except sr.RequestError as exc:
            raise VoiceError(
                f"Speech recognition service unavailable: {exc}"
            ) from exc

    def shutdown(self) -> None:
        """Stop playback and cleanly terminate the TTS worker thread."""
        self.stop_speaking()
        self._stop_event.set()
        self._tts_queue.put(None)
        try:
            if pygame.mixer.get_init():
                pygame.mixer.quit()
        except Exception:  # noqa: BLE001
            pass