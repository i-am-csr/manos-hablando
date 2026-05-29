import threading

from loguru import logger


class LetterBuffer:
    """
    Debounced letter accumulator for fingerspelling recognition.

    Frames stream in faster than letters change, so we need to wait until a
    letter prediction is stable for a number of consecutive frames before
    committing it to the current word. Pauses (no hand visible) act as word
    boundaries.

    Thread-safe — designed for use from a streamlit-webrtc worker thread.
    """

    def __init__(
        self,
        stability_window: int = 15,
        confidence_threshold: float = 0.7,
        word_boundary_frames: int = 30,
    ):
        """
        Args:
            stability_window: Number of consecutive frames a letter must be
                predicted before it is committed to the buffer.
            confidence_threshold: Minimum confidence for a prediction to
                count toward the stability window.
            word_boundary_frames: Number of consecutive no-hand frames that
                signal the end of a word (default 30 ≈ 1s at 30fps).
        """
        self.stability_window = stability_window
        self.confidence_threshold = confidence_threshold
        self.word_boundary_frames = word_boundary_frames

        self._lock = threading.Lock()
        self._letters: list[str] = []
        self._candidate: str | None = None
        self._candidate_count: int = 0
        self._candidate_committed: bool = False
        self._no_hand_count: int = 0
        self._word_complete: bool = False

    def add_prediction(self, letter: str, confidence: float) -> None:
        """
        Register a per-frame prediction.

        A letter is only committed to the current word after it has been
        predicted with sufficient confidence for `stability_window`
        consecutive frames. Repeated same-letter frames after commit are
        ignored until a different letter breaks the streak.
        """
        with self._lock:
            self._no_hand_count = 0

            if confidence < self.confidence_threshold:
                # Low-confidence frame breaks the streak without resetting
                # the word boundary timer.
                self._reset_candidate()
                return

            if letter == self._candidate:
                self._candidate_count += 1
            else:
                self._candidate = letter
                self._candidate_count = 1
                self._candidate_committed = False

            if (
                self._candidate_count >= self.stability_window
                and not self._candidate_committed
            ):
                self._letters.append(letter)
                self._candidate_committed = True
                logger.debug(
                    f"Letter committed: '{letter}' "
                    f"(word so far: {''.join(self._letters)})"
                )

    def add_no_hand(self) -> None:
        """Register a frame where no hand was detected."""
        with self._lock:
            self._reset_candidate()
            self._no_hand_count += 1

            if (
                self._no_hand_count >= self.word_boundary_frames
                and self._letters
                and not self._word_complete
            ):
                self._word_complete = True
                logger.info(
                    f"Word boundary detected: '{''.join(self._letters)}'"
                )

    def get_current_word(self) -> str:
        """Return the letters committed so far, concatenated."""
        with self._lock:
            return "".join(self._letters)

    def word_complete(self) -> bool:
        """Return True if a pause has marked the end of a word."""
        with self._lock:
            return self._word_complete

    def reset(self) -> None:
        """Clear all state — typically called after the word is consumed."""
        with self._lock:
            self._letters = []
            self._reset_candidate()
            self._no_hand_count = 0
            self._word_complete = False

    def _reset_candidate(self) -> None:
        """Internal: clear the in-progress candidate (must hold the lock)."""
        self._candidate = None
        self._candidate_count = 0
        self._candidate_committed = False
