"""
Inference errors.

``ProviderError`` carries the upstream status code when one is
available, so the runner's retry policy can tell a rate limit from a
malformed request without knowing anything about the provider.
"""


class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        status_code: int | None = None,
    ):
        super().__init__(message)

        self.status_code = status_code
