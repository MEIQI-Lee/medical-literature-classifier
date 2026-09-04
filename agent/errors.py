class AgentError(Exception):
    """An expected failure with a stable machine-readable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
