from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from voyager_ai.retry import RetryableError, parse_retry_after, retry_call


def _always(exc):
    def fn():
        raise exc

    return fn


def test_retries_then_succeeds():
    sleeps, calls = [], iter([RetryableError("x"), RetryableError("x"), "ok"])

    def fn():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    assert retry_call(fn, sleep=sleeps.append, random=lambda: 0) == "ok"
    assert sleeps == [1.0, 2.0]


def test_honours_retry_after_and_caps_it():
    for hint, expected in ((7, 7.0), (120, 30.0)):
        sleeps = []
        with pytest.raises(RetryableError):
            retry_call(
                _always(RetryableError("429", retry_after=hint)),
                attempts=2,
                sleep=sleeps.append,
                random=lambda: 0,
            )
        assert sleeps == [expected]


def test_other_errors_are_not_retried():
    fn = MagicMock(side_effect=ValueError("bad"))
    with pytest.raises(ValueError):
        retry_call(fn, sleep=lambda s: None)
    assert fn.call_count == 1


def test_gives_up_after_attempts():
    fn = MagicMock(side_effect=RetryableError("x"))
    with pytest.raises(RetryableError):
        retry_call(fn, attempts=3, sleep=lambda s: None)
    assert fn.call_count == 3


def test_parse_retry_after():
    now = datetime(2026, 10, 21, 7, 28, 0, tzinfo=timezone.utc)
    assert parse_retry_after("3") == 3.0
    assert parse_retry_after("Wed, 21 Oct 2026 07:28:10 GMT", now=now) == 10.0
    assert parse_retry_after("soon") is None
    assert parse_retry_after(None) is None
