"""Runner and recorder lifecycle states plus injectable clocks."""

from enum import Enum
import time


class RunnerState(str, Enum):
    CREATED = "created"
    CONNECTED = "connected"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CLOSED = "closed"


class RecorderState(str, Enum):
    CREATED = "created"
    RECORDING = "recording"
    FINALIZED = "finalized"
    CLOSED = "closed"


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    def wall_time_utc_ns(self) -> int:
        return time.time_ns()


class DeterministicClock:
    def __init__(self, start: int = 0, increment: int = 1):
        self.value = start
        self.increment = increment

    def monotonic_ns(self) -> int:
        value = self.value
        self.value += self.increment
        return value

    def wall_time_utc_ns(self) -> int:
        return self.monotonic_ns()
