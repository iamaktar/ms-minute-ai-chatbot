"""Desktop GUI with streaming chat, live model switching, and multilingual voice I/O for Ms. Minute."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk

from chatbot import ChatbotError, GeminiChatbot, NetworkTimeoutError, RateLimitError
from config import ConfigurationError, Settings
from voice import DEFAULT_VOICE_LABEL, VOICE_OPTIONS, VoiceEngine, VoiceError

BOT_NAME = "Ms. Minute"

AVAILABLE_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]


class MsMinuteApp(ctk.CTk):
    """Main CustomTkinter application window for Ms. Minute."""

    def __init__(self, bot: GeminiChatbot) -> None:
        super().__init__()
        self._bot = bot
        self._voice = VoiceEngine(voice_label=DEFAULT_VOICE_LABEL)
        self._is_busy = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"{BOT_NAME} — AI Voice Assistant")
        self.geometry("1040x720")
        self.minsize(800, 580)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()
        self._append_system_message(
            f"{BOT_NAME} is ready ({self._bot.model_name}). "
            "Switch models or voices on the left, type a message, or click Speak."
        )

    def _build_layout(self) -> None:
        """Construct the sidebar, chat transcript, and input controls."""
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar ---
        sidebar = ctk.CTkFrame(self, width=270, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(10, weight=1)

        title_label = ctk.CTkLabel(
            sidebar,
            text=BOT_NAME,
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.grid(row=0, column=0, padx=20, pady=(20, 12), sticky="w")

        # Model Selector
        model_label = ctk.CTkLabel(
            sidebar,
            text="Gemini Model",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        model_label.grid(row=1, column=0, padx=20, pady=(0, 6), sticky="w")

        model_choices = list(AVAILABLE_MODELS)
        if self._bot.model_name not in model_choices:
            model_choices.insert(0, self._bot.model_name)

        self._model_menu = ctk.CTkOptionMenu(
            sidebar,
            values=model_choices,
            command=self._handle_model_change,
        )
        self._model_menu.set(self._bot.model_name)
        self._model_menu.grid(row=2, column=0, padx=20, pady=(0, 14), sticky="ew")

        # Voice Selector
        voice_label = ctk.CTkLabel(
            sidebar,
            text="Voice & Language",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        voice_label.grid(row=3, column=0, padx=20, pady=(0, 6), sticky="w")

        self._voice_menu = ctk.CTkOptionMenu(
            sidebar,
            values=list(VOICE_OPTIONS.keys()),
            command=self._handle_voice_change,
        )
        self._voice_menu.set(DEFAULT_VOICE_LABEL)
        self._voice_menu.grid(row=4, column=0, padx=20, pady=(0, 8), sticky="ew")

        test_voice_btn = ctk.CTkButton(
            sidebar,
            text="Test Voice",
            fg_color="#2563EB",
            hover_color="#1D4ED8",
            command=self._handle_test_voice,
        )
        test_voice_btn.grid(row=5, column=0, padx=20, pady=(0, 12), sticky="ew")

        self._auto_speak_var = ctk.BooleanVar(value=True)
        auto_speak_switch = ctk.CTkSwitch(
            sidebar,
            text="Auto-Speak Replies",
            variable=self._auto_speak_var,
        )
        auto_speak_switch.grid(row=6, column=0, padx=20, pady=(0, 10), sticky="w")

        self._stop_voice_btn = ctk.CTkButton(
            sidebar,
            text="Stop Speaking",
            fg_color="#B45309",
            hover_color="#92400E",
            command=self._handle_stop_speaking,
        )
        self._stop_voice_btn.grid(row=7, column=0, padx=20, pady=(0, 16), sticky="ew")

        # Persona Editor
        persona_label = ctk.CTkLabel(
            sidebar,
            text="System Persona",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        persona_label.grid(row=8, column=0, padx=20, pady=(0, 6), sticky="w")

        self._persona_box = ctk.CTkTextbox(sidebar, height=100, wrap="word")
        self._persona_box.grid(row=9, column=0, padx=20, pady=(0, 8), sticky="ew")
        self._persona_box.insert("1.0", self._bot.system_instruction)

        update_persona_btn = ctk.CTkButton(
            sidebar,
            text="Update Persona",
            command=self._handle_update_persona,
        )
        update_persona_btn.grid(row=10, column=0, padx=20, pady=(0, 12), sticky="new")

        clear_btn = ctk.CTkButton(
            sidebar,
            text="Clear Chat",
            fg_color="#374151",
            hover_color="#4B5563",
            command=self._handle_clear_chat,
        )
        clear_btn.grid(row=11, column=0, padx=20, pady=(0, 12), sticky="ew")

        self._status_label = ctk.CTkLabel(
            sidebar,
            text="Status: Ready",
            font=ctk.CTkFont(size=12),
            text_color="#34D399",
        )
        self._status_label.grid(row=12, column=0, padx=20, pady=(0, 18), sticky="w")

        # --- Main Chat Area ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=1, sticky="nsew", padx=16, pady=16)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self._chat_box = ctk.CTkTextbox(
            main_frame,
            wrap="word",
            font=ctk.CTkFont(size=14),
            state="disabled",
        )
        self._chat_box.grid(row=0, column=0, sticky="nsew", pady=(0, 12))
        self._chat_box.tag_config("user_label", foreground="#60A5FA")
        self._chat_box.tag_config("bot_label", foreground="#34D399")
        self._chat_box.tag_config("system_msg", foreground="#9CA3AF")
        self._chat_box.tag_config("error_msg", foreground="#F87171")

        # --- Input Bar ---
        input_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        input_frame.grid(row=1, column=0, sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self._input_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text=f"Ask {BOT_NAME} anything...",
            height=42,
            font=ctk.CTkFont(size=14),
        )
        self._input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self._input_entry.bind("<Return>", lambda _event: self._handle_send_text())

        self._speak_btn = ctk.CTkButton(
            input_frame,
            text="Speak",
            width=95,
            height=42,
            fg_color="#059669",
            hover_color="#047857",
            command=self._handle_voice_input,
        )
        self._speak_btn.grid(row=0, column=1, padx=(0, 8))

        self._send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            width=95,
            height=42,
            command=self._handle_send_text,
        )
        self._send_btn.grid(row=0, column=2)

    def _set_status(self, text: str, color: str = "#34D399") -> None:
        self.after(
            0,
            lambda: self._status_label.configure(
                text=f"Status: {text}", text_color=color
            ),
        )

    def _set_busy(self, busy: bool) -> None:
        self._is_busy = busy
        state = "disabled" if busy else "normal"
        self.after(0, lambda: self._send_btn.configure(state=state))
        self.after(0, lambda: self._speak_btn.configure(state=state))
        self.after(0, lambda: self._input_entry.configure(state=state))
        self.after(0, lambda: self._model_menu.configure(state=state))

    def _insert_chat(self, text: str, tag: str | None = None) -> None:
        self._chat_box.configure(state="normal")
        if tag:
            self._chat_box.insert(tk.END, text, tag)
        else:
            self._chat_box.insert(tk.END, text)
        self._chat_box.see(tk.END)
        self._chat_box.configure(state="disabled")

    def _append_system_message(self, message: str) -> None:
        self._insert_chat(f"[System] {message}\n\n", "system_msg")

    def _append_error_message(self, message: str) -> None:
        self._insert_chat(f"[Error] {message}\n\n", "error_msg")

    def _handle_model_change(self, selected_model: str) -> None:
        self._bot.set_model_name(selected_model)
        self._append_system_message(f"Switched active model to {selected_model}.")

    def _handle_voice_change(self, selected_label: str) -> None:
        voice = self._voice.set_voice(selected_label)
        self._append_system_message(
            f"Voice switched to {voice.label} (Mic language: {voice.stt_language})."
        )

    def _handle_test_voice(self) -> None:
        self._voice.speak_sample()

    def _handle_stop_speaking(self) -> None:
        self._voice.stop_speaking()
        if not self._is_busy:
            self._set_status("Ready")

    def _handle_update_persona(self) -> None:
        new_persona = self._persona_box.get("1.0", tk.END).strip()
        if not new_persona:
            self._append_error_message("System persona cannot be empty.")
            return
        self._bot.set_system_instruction(new_persona)
        self._append_system_message("System persona updated.")

    def _handle_clear_chat(self) -> None:
        if self._is_busy:
            return
        self._voice.stop_speaking()
        self._bot.clear_history()
        self._chat_box.configure(state="normal")
        self._chat_box.delete("1.0", tk.END)
        self._chat_box.configure(state="disabled")
        self._append_system_message("Conversation memory cleared.")

    def _handle_send_text(self) -> None:
        if self._is_busy:
            return
        user_text = self._input_entry.get().strip()
        if not user_text:
            return
        self._input_entry.delete(0, tk.END)
        self._start_reply_thread(user_text)

    def _handle_voice_input(self) -> None:
        if self._is_busy:
            return
        self._voice.stop_speaking()
        self._set_busy(True)
        self._set_status("Listening...", "#FBBF24")

        def worker() -> None:
            try:
                spoken_text = self._voice.listen()
            except VoiceError as exc:
                self.after(0, lambda msg=str(exc): self._append_error_message(msg))
                self._set_status("Ready")
                self._set_busy(False)
                return

            self.after(
                0,
                lambda text=spoken_text: self._start_reply_thread(
                    text, from_voice=True
                ),
            )

        threading.Thread(target=worker, daemon=True).start()

    def _start_reply_thread(self, user_text: str, from_voice: bool = False) -> None:
        """Stream Ms. Minute's response in a background thread."""
        self._voice.stop_speaking()
        self._set_busy(True)
        self._set_status("Thinking...", "#60A5FA")

        prefix = "You (Voice): " if from_voice else "You: "
        self._insert_chat(prefix, "user_label")
        self._insert_chat(f"{user_text}\n\n")
        self._insert_chat(f"{BOT_NAME}: ", "bot_label")

        def worker() -> None:
            chunks: list[str] = []
            try:
                for chunk in self._bot.stream_reply(user_text):
                    chunks.append(chunk)
                    self.after(0, lambda c=chunk: self._insert_chat(c))
                self.after(0, lambda: self._insert_chat("\n\n"))

                full_reply = "".join(chunks).strip()
                if full_reply and self._auto_speak_var.get():
                    self._set_status("Speaking...", "#A78BFA")
                    self._voice.speak(full_reply)
            except (RateLimitError, NetworkTimeoutError, ChatbotError) as exc:
                self.after(
                    0,
                    lambda msg=str(exc): self._insert_chat(
                        f"\n[Error] {msg}\n\n", "error_msg"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(
                    0,
                    lambda msg=str(exc): self._insert_chat(
                        f"\n[Unexpected error] {msg}\n\n", "error_msg"
                    ),
                )
            finally:
                self._set_status("Ready")
                self._set_busy(False)
                self.after(0, lambda: self._input_entry.focus_set())

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        """Cleanly shut down voice threads before closing the window."""
        self._voice.shutdown()
        self.destroy()


def main() -> int:
    """Launch the Ms. Minute desktop GUI."""
    try:
        settings = Settings.from_env()
    except ConfigurationError as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Configuration Error", str(exc))
        root.destroy()
        return 1

    bot = GeminiChatbot(settings=settings)
    app = MsMinuteApp(bot=bot)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())