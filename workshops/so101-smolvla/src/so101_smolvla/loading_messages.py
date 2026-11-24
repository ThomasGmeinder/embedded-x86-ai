"""What loading a model says: plain lines, no progress bar, no notice that is not this workshop's to act on.

Standard library only, and no import of the workshop's package: the SAM 3 worker
runs in its own environment, where the package is not installed, and imports
this file as the module beside it. The main process imports it from the package.

In a notebook a library's progress bar and its deprecation notices come out on
standard error and are shown as if something had gone wrong. The two helpers
here change what is said, not what is done.
"""

from __future__ import annotations

import logging


def plain_progress(items, desc: str = "", **_):
    """In place of a progress bar: the items, and one line on standard output when all are through."""
    count = 0
    for item in items:
        count += 1
        yield item
    print(f"{desc}: {count} of {count}", flush=True)


def loading_weights_as_a_line() -> bool:
    """Have transformers say "Loading weights: N of N" in one line instead of drawing a bar.

    transformers loads a model's weights in one loop under a progress bar
    (``core_model_loading``, its name ``tqdm``). Whether there was such a bar
    to replace; a transformers without it is left as it is.
    """
    from transformers import core_model_loading

    if not hasattr(core_model_loading, "tqdm"):
        return False
    core_model_loading.tqdm = plain_progress
    return True


class WithoutNotice(logging.Filter):
    """Lets every record through but the ones that hold ``text``."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def filter(self, record: logging.LogRecord) -> bool:
        return self.text not in record.getMessage()


def drop_transformers_notice(text: str) -> None:
    """Keep one notice of transformers from being printed, whichever of its modules logs it.

    For a notice about a call this workshop does not make itself (LeRobot's
    ``torch_dtype`` for SmolVLA's language model): nothing here can act on it.
    """
    from transformers.utils import logging as transformers_logging

    for handler in transformers_logging.get_logger().handlers:
        if not any(isinstance(f, WithoutNotice) and f.text == text for f in handler.filters):
            handler.addFilter(WithoutNotice(text))
