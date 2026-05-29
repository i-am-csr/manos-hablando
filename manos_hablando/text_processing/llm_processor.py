import os
import threading
import time
import traceback

# Use the OS trust store (macOS keychain, Windows cert store, Linux ca-certificates)
# instead of certifi. Required when behind corporate SSL inspection proxies
# (e.g. Cisco Secure Access) whose CA is trusted by the OS but not by certifi.
# Must run BEFORE httpx/openai create any SSL context.
import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
from loguru import logger
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

SYSTEM_PROMPT = """You are a Mexican Sign Language (LSM) fingerspelling interpreter.
The user fingerspelled a sequence of letters that may contain recognition errors.
Your job is to reconstruct the most likely Spanish content the user signed.

The content may be a common Spanish word, a phrase, OR a proper noun
(a person's first name, surname, place name, brand, etc.). Do not assume
it must be a dictionary word — names are equally valid.

Common recognition error patterns to consider when reconstructing:
- Closed-fist family: A ↔ M ↔ N ↔ S ↔ T (often interchangeable)
- Pointing-finger family: R ↔ V ↔ U ↔ K (often interchangeable)
- Long-frame artifacts: doubled letters (e.g. "AA") may be a single letter
- Transition noise: extra letters between real ones may be spurious

If a CONTEXT HINT is provided in the user message, it tells you what kind
of content to expect (e.g. "Spanish first name and surnames"). Weight your
reconstruction toward that domain.

Rules:
- Return ONLY the reconstructed content, nothing else
- Preserve word boundaries if the input has clear spaces or pauses
- If the sequence is too ambiguous, return your best guess
- Never return an empty string"""


class LLMProcessor:
    """
    Reconstructs Spanish words from noisy fingerspelled letter sequences
    using a LangChain chat model.

    Thread-safe — designed for use from a streamlit-webrtc worker thread.
    Falls back to the raw letters joined if the model invocation fails.

    Exposes `last_status` so the UI can show whether the most recent call
    succeeded, fell back, or is still in flight:
        "idle"     — no call yet
        "calling"  — request in flight
        "success"  — LLM returned a reconstruction
        "fallback" — error, returned raw letters
        "no_key"   — API key missing, returned raw letters
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        model_provider: str = "openai",
        temperature: float = 0.3,
        max_tokens: int = 64,
        timeout: float = 15.0,
    ):
        """
        Args:
            model: Model identifier (e.g. "gpt-4o", "gpt-4o-mini").
            model_provider: LangChain provider name (e.g. "openai").
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.
            timeout: Per-request timeout in seconds.
        """
        self.model_name = model
        self.provider = model_provider

        # Public status fields the UI can read
        self.last_status: str = "idle"
        self.last_input: list[str] = []
        self.last_output: str = ""
        self.last_error: str = ""
        self.last_latency_ms: float = 0.0
        self.last_call_at: float = 0.0
        self.call_count: int = 0

        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OPENAI_API_KEY not set — LLM reconstruction will fall back "
                "to raw letter sequences."
            )
            self._chat = None
            self.last_status = "no_key"
        else:
            logger.info(
                f"Initializing LLM: provider={model_provider} model={model} "
                f"timeout={timeout}s"
            )
            self._chat = init_chat_model(
                model=model,
                model_provider=model_provider,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        self._lock = threading.Lock()

    def reconstruct(
        self,
        letters: list[str],
        context: str | None = None,
    ) -> str:
        """
        Send a letter sequence to the chat model and return the reconstructed
        word.

        Args:
            letters: per-frame letter predictions in time order.
            context: optional one-line hint about what kind of content this is
                (e.g. "a Spanish person's first name and surnames",
                "a single common Spanish word"). When provided, it's injected
                into the human message to bias the LLM appropriately.

        Falls back to the raw letters joined together on any failure
        (missing API key, network error, empty response, etc.).
        """
        if not letters:
            return ""

        raw = "".join(letters)
        self.last_input = list(letters)
        self.last_call_at = time.time()
        self.call_count += 1

        if self._chat is None:
            self.last_status = "no_key"
            self.last_output = raw
            self.last_error = "OPENAI_API_KEY not set"
            logger.warning(
                f"[LLM #{self.call_count}] No API key — returning raw '{raw}'"
            )
            return raw

        human_lines = [f"Fingerspelled letters: {' '.join(letters)}"]
        if context:
            human_lines.append(f"Context hint: {context}")
        human_lines.append("Reconstruct the most likely content.")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="\n".join(human_lines)),
        ]

        self.last_status = "calling"
        logger.info(
            f"[LLM #{self.call_count}] Calling {self.provider}:{self.model_name} "
            f"with letters {letters}..."
        )
        start = time.perf_counter()

        try:
            with self._lock:
                response = self._chat.invoke(messages)
            latency_ms = (time.perf_counter() - start) * 1000
            self.last_latency_ms = latency_ms

            text = (response.content or "").strip()
            if not text:
                self.last_status = "fallback"
                self.last_output = raw
                self.last_error = "empty response"
                logger.warning(
                    f"[LLM #{self.call_count}] Empty response in "
                    f"{latency_ms:.0f}ms, using raw '{raw}'"
                )
                return raw

            self.last_status = "success"
            self.last_output = text
            self.last_error = ""
            logger.success(
                f"[LLM #{self.call_count}] {letters} → '{text}' "
                f"in {latency_ms:.0f}ms"
            )
            return text

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            self.last_latency_ms = latency_ms
            self.last_status = "fallback"
            self.last_output = raw
            err_type = type(e).__name__
            err_msg = str(e) or "no message"
            self.last_error = f"{err_type}: {err_msg}"
            logger.error(
                f"[LLM #{self.call_count}] {err_type} after "
                f"{latency_ms:.0f}ms: {err_msg}"
            )
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            return raw
