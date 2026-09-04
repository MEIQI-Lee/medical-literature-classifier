from pathlib import Path
from threading import Lock

from agent.schemas import AgentResult


class JsonlTraceSink:
    """Append final public results without storing raw document text.

    The lock coordinates threads in one process. Use one log file per worker process.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, result: AgentResult):
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")
