"""Streamlit web interface for Ms. Minute AI Chatbot."""

from __future__ import annotations

import asyncio
import base64
import os
import re
from typing import Any

import edge_tts
import streamlit as st

from chatbot import ChatbotError, GeminiChatbot, NetworkTimeoutError, RateLimitError
from config import ConfigurationError, Settings

BOT_NAME = "Ms. Minute"

AVAILABLE_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]

VOICE_OPTIONS: dict[str, tuple[str, str]] = {
    "English (India) - Neerja": (
        "en-IN-NeerjaNeural",
        "Hello! I am Ms. Minute, and my voice is working.",
    ),
    "English (US) - Aria": (
        "en-US-AriaNeural",
        "Hello! I am Ms. Minute, ready to help you.",
    ),
    "English (UK) - Sonia": (
        "en-GB-SoniaNeural",
        "Hello! I am Ms. Minute, at your service.",
    ),
    "Bengali (India) - Tanishaa": (
        "bn-IN-TanishaaNeural",
        "নমস্কার! আমি মিস মিনিট, এখন আমি বাংলায় কথা বলতে পারি।",
    ),
    "Hindi (India) - Swara": (
        "hi-IN-SwaraNeural",
        "नमस्ते! मैं मिस मिनट हूँ, और अब मैं हिंदी में बात कर सकती हूँ।",
    ),
}

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


async def _synthesize_mp3_bytes(text: str, voice: str) -> bytes:
    """Synthesize speech to in-memory MP3 bytes using Edge Neural TTS."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


def generate_speech_bytes(
    text: str,
    preferred_voice: str,
    auto_detect: bool = True,
) -> bytes | None:
    """Return MP3 audio bytes for the given text, or None if synthesis fails."""
    spoken_text = clean_text_for_speech(text)
    if not spoken_text:
        return None
    selected_voice = (
        detect_tts_voice(spoken_text, preferred_voice)
        if auto_detect
        else preferred_voice
    )
    try:
        return asyncio.run(_synthesize_mp3_bytes(spoken_text, selected_voice))
    except Exception:  # noqa: BLE001
        return None


def play_hidden_audio(audio_bytes: bytes) -> None:
    """Autoplay MP3 audio in the background without showing an audio player bar."""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    st.markdown(
        f'<audio autoplay="true" style="display:none;">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f"</audio>",
        unsafe_allow_html=True,
    )


def load_settings() -> Settings:
    """Load settings from Streamlit Cloud secrets or local .env."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            os.environ["GEMINI_API_KEY"] = str(st.secrets["GEMINI_API_KEY"])
        for optional_key in (
            "GEMINI_MODEL",
            "GEMINI_TEMPERATURE",
            "GEMINI_SYSTEM_INSTRUCTION",
            "MAX_HISTORY_MESSAGES",
            "REQUEST_TIMEOUT_SECONDS",
        ):
            if optional_key in st.secrets:
                os.environ[optional_key] = str(st.secrets[optional_key])
    except Exception:  # noqa: BLE001
        pass

    return Settings.from_env()


def init_session_state() -> None:
    """Initialize the chatbot instance and UI chat history in session state."""
    if "bot" not in st.session_state:
        settings = load_settings()
        st.session_state.bot = GeminiChatbot(settings=settings)
    if "ui_messages" not in st.session_state:
        st.session_state.ui_messages: list[dict[str, Any]] = []


def main() -> None:
    """Render the Streamlit web application."""
    st.set_page_config(
        page_title=f"{BOT_NAME} — AI Assistant",
        page_icon="🤖",
        layout="wide",
    )

    try:
        init_session_state()
    except ConfigurationError as exc:
        st.error(f"Configuration Error: {exc}")
        st.info(
            "On Streamlit Cloud, go to **App Settings → Secrets** and add:\n\n"
            '`GEMINI_API_KEY = "your_api_key_here"`'
        )
        st.stop()

    bot: GeminiChatbot = st.session_state.bot

    # --- Sidebar Controls ---
    with st.sidebar:
        st.title(f"🤖 {BOT_NAME}")
        st.caption("Multilingual AI Chat, Voice & Visual Assistant")

        model_choices = list(AVAILABLE_MODELS)
        if bot.model_name not in model_choices:
            model_choices.insert(0, bot.model_name)

        selected_model = st.selectbox(
            "Gemini Model",
            options=model_choices,
            index=model_choices.index(bot.model_name),
        )
        if selected_model != bot.model_name:
            bot.set_model_name(selected_model)
            st.toast(f"Switched model to {selected_model}")

        selected_voice_label = st.selectbox(
            "Preferred Voice (Auto-detects EN / BN / HI)",
            options=list(VOICE_OPTIONS.keys()),
            index=0,
        )
        preferred_voice, sample_text = VOICE_OPTIONS[selected_voice_label]

        if st.button("🔊 Test Voice", use_container_width=True):
            audio_bytes = generate_speech_bytes(
                sample_text, preferred_voice, auto_detect=False
            )
            if audio_bytes:
                play_hidden_audio(audio_bytes)
            else:
                st.warning("Could not play voice sample.")

        auto_speak = st.toggle("Auto-Speak Replies", value=True)

        st.divider()

        if st.button("Clear Chat", type="secondary", use_container_width=True):
            bot.clear_history()
            st.session_state.ui_messages = []
            st.rerun()

    # --- Main Chat Interface ---
    st.header(BOT_NAME)

    for idx, msg in enumerate(st.session_state.ui_messages):
        with st.chat_message(msg["role"]):
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("image"):
                st.image(
                    msg["image"],
                    caption=msg.get("caption", ""),
                    use_container_width=True,
                )
                st.download_button(
                    label="Download Image",
                    data=msg["image"],
                    file_name=f"ms_minute_{idx}.png",
                    mime="image/png",
                    key=f"dl_{idx}",
                )

    prompt = st.chat_input(f"Ask {BOT_NAME} anything...")
    if not prompt:
        return

    # Display user message
    st.session_state.ui_messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Handle /image command
    if prompt.strip().lower().startswith("/image "):
        image_prompt = prompt.strip()[7:].strip()
        with st.chat_message("assistant"):
            with st.spinner("Generating image..."):
                try:
                    img_bytes = bot.generate_image(image_prompt)
                    st.image(img_bytes, caption=image_prompt, use_container_width=True)
                    st.download_button(
                        label="Download Image",
                        data=img_bytes,
                        file_name="ms_minute_generated.png",
                        mime="image/png",
                    )
                    st.session_state.ui_messages.append(
                        {
                            "role": "assistant",
                            "text": f"Generated image for: *{image_prompt}*",
                            "image": img_bytes,
                            "caption": image_prompt,
                        }
                    )
                except (RateLimitError, NetworkTimeoutError, ChatbotError) as exc:
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Unexpected error: {exc}")
        return

    # Handle normal streaming text reply
    with st.chat_message("assistant"):
        try:
            full_reply = st.write_stream(bot.stream_reply(prompt))
            if isinstance(full_reply, str) and full_reply.strip() and auto_speak:
                audio_bytes = generate_speech_bytes(full_reply, preferred_voice)
                if audio_bytes:
                    play_hidden_audio(audio_bytes)

            st.session_state.ui_messages.append(
                {
                    "role": "assistant",
                    "text": str(full_reply),
                }
            )
        except (RateLimitError, NetworkTimeoutError, ChatbotError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")


if __name__ == "__main__":
    main()