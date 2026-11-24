"""loading_messages: a line instead of a progress bar, and one library notice kept from being printed."""

from __future__ import annotations

import logging

from so101_smolvla import loading_messages
from so101_smolvla.loading_messages import WithoutNotice, drop_transformers_notice, plain_progress


def test_the_plain_progress_hands_every_item_on_and_says_one_line_when_all_are_through(capsys):
    seen = []
    for item in plain_progress({"a": 1, "b": 2, "c": 3}.items(), desc="Loading weights"):
        seen.append(item)
        assert capsys.readouterr().out == ""  # nothing while the loop runs
    assert seen == [("a", 1), ("b", 2), ("c", 3)]
    said = capsys.readouterr()
    assert said.out == "Loading weights: 3 of 3\n" and said.err == ""


def test_transformers_loads_weights_under_the_plain_progress_once_asked():
    from transformers import core_model_loading

    before = core_model_loading.tqdm
    try:
        assert loading_messages.loading_weights_as_a_line() is True
        assert core_model_loading.tqdm is plain_progress
    finally:
        core_model_loading.tqdm = before


def test_one_notice_is_dropped_and_every_other_record_goes_through():
    notice = WithoutNotice("`torch_dtype` is deprecated")

    def record(text: str) -> logging.LogRecord:
        return logging.LogRecord("transformers.modeling_utils", logging.WARNING, __file__, 1, text, None, None)

    assert notice.filter(record("`torch_dtype` is deprecated! Use `dtype` instead!")) is False
    assert notice.filter(record("Some weights were not initialized")) is True


def test_the_notice_filter_sits_on_transformers_handlers_once():
    from transformers.utils import logging as transformers_logging

    handlers = transformers_logging.get_logger().handlers
    assert handlers  # transformers prints through its own handler
    before = [list(handler.filters) for handler in handlers]
    try:
        drop_transformers_notice("a notice of this test")
        drop_transformers_notice("a notice of this test")  # asked twice, added once
        for handler in handlers:
            ours = [f for f in handler.filters if isinstance(f, WithoutNotice) and f.text == "a notice of this test"]
            assert len(ours) == 1
    finally:
        for handler, filters in zip(handlers, before):
            handler.filters[:] = filters
