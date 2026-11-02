"""Keep the model's context small: bounded command output and masked old tool results."""

import codecs
from asyncio import StreamReader

from openai.types.responses.response_input_item_param import FunctionCallOutput

HEAD_CHARS = 3_000
TAIL_CHARS = 7_000
MAX_OUTPUT_CHARS = HEAD_CHARS + TAIL_CHARS
KEEP_RECENT_RESULTS = 10
MASKED_RESULT = "[output omitted to save context; rerun the tool if you need it]"
READ_BYTES = 65_536


def _omission(count: int) -> str:
    return f"\n[... {count:,} characters omitted ...]\n"


def head_and_tail(text: str) -> str:
    """Cut the middle out of `text` once it passes MAX_OUTPUT_CHARS, keeping its start and end."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:HEAD_CHARS] + _omission(len(text) - MAX_OUTPUT_CHARS) + text[-TAIL_CHARS:]


class Capture:
    """Reads a stream to its end but keeps only the head and tail, so a flood can't fill memory."""

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._head = ""
        self._tail = ""
        self._omitted = 0

    @property
    def text(self) -> str:
        """Everything read so far, with a marker where the middle was dropped."""
        middle = _omission(self._omitted) if self._omitted else ""
        return self._head + middle + self._tail

    async def read(self, stream: StreamReader) -> None:
        while data := await stream.read(READ_BYTES):
            self._add(self._decoder.decode(data))
        self._add(self._decoder.decode(b"", final=True))

    def _add(self, text: str) -> None:
        room = HEAD_CHARS - len(self._head)
        self._head += text[:room]
        self._tail += text[room:]
        if len(self._tail) > TAIL_CHARS:
            self._omitted += len(self._tail) - TAIL_CHARS
            self._tail = self._tail[-TAIL_CHARS:]


def mask_old_results(results: list[FunctionCallOutput]) -> None:
    """Replace all but the newest KEEP_RECENT_RESULTS tool results; the calls stay visible."""
    for result in results[:-KEEP_RECENT_RESULTS]:
        result["output"] = MASKED_RESULT
