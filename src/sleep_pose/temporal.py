"""Hysteresis-based temporal smoothing for noisy frame-level pose labels."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable

from .models import Detection


class PoseSmoother:
    def __init__(
        self,
        *,
        window_size: int = 5,
        minimum_votes: int = 3,
        transition_frames: int = 2,
        confidence_threshold: float = 0.4,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= minimum_votes <= window_size:
            raise ValueError("minimum_votes must be within the window")
        if transition_frames < 1:
            raise ValueError("transition_frames must be positive")
        self.history: deque[str | None] = deque(maxlen=window_size)
        self.minimum_votes = minimum_votes
        self.transition_frames = transition_frames
        self.confidence_threshold = confidence_threshold
        self.current: str | None = None
        self.candidate: str | None = None
        self.candidate_streak = 0

    def update(self, detections: Iterable[Detection]) -> str | None:
        eligible = [
            detection
            for detection in detections
            if detection.confidence >= self.confidence_threshold
        ]
        observed = max(eligible, key=lambda item: item.confidence).label if eligible else None
        self.history.append(observed)
        votes = Counter(item for item in self.history if item is not None)
        leading = max(votes, key=lambda item: (votes[item], item)) if votes else None

        if leading is None or votes[leading] < self.minimum_votes:
            self.candidate = None
            self.candidate_streak = 0
            return self.current
        if leading == self.current:
            self.candidate = None
            self.candidate_streak = 0
            return self.current
        if leading == self.candidate:
            self.candidate_streak += 1
        else:
            self.candidate = leading
            self.candidate_streak = 1
        if self.candidate_streak >= self.transition_frames:
            self.current = leading
            self.candidate = None
            self.candidate_streak = 0
        return self.current
