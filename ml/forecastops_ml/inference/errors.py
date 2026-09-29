"""Public failures for batch inference.

Messages returned to callers are short English sentences. They do not include
observation rows from the dataset.
"""

_GENERIC = "Batch inference failed."


class InferenceError(Exception):
    """A forecast cannot continue.

    The message is safe to store on the forecast run.
    """


class BatchCapacityError(InferenceError):
    """The daily batch inference ceiling has been reached."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"The daily batch inference limit of {limit} jobs has been reached.")


def public_inference_error(exc: BaseException) -> str:
    """Return an English message safe to store on a forecast run."""

    message = " ".join(str(exc).split())
    if _is_public_message(message):
        return message
    return _GENERIC


def sanitize_failure_reason(reason: object) -> str:
    """Keep a short English failure reason and drop anything else."""

    if isinstance(reason, str):
        message = " ".join(reason.split())
        if _is_public_message(message):
            return message
    return _GENERIC


def _is_public_message(message: str) -> bool:
    if message == "" or len(message) > 200:
        return False
    if not message.isascii():
        return False
    if any(char in message for char in "{}[]"):
        return False
    if message.count(",") > 3:
        return False
    digits = sum(char.isdigit() for char in message)
    if digits > 12:
        return False
    letters = sum(char.isalpha() for char in message)
    return letters >= 8
