"""Unit tests for the helpers inside execution.py that don't need a kernel."""

from __future__ import annotations

import nbformat

from mcp_jupyter_driver.execution import (
    MAX_OUTPUT_BYTES,
    _cap_output,
    _coalesce_stream,
    _output_size,
    _truncate_str,
)


def test_truncate_short_string_unchanged() -> None:
    assert _truncate_str("hello") == "hello"


def test_truncate_long_string_marked() -> None:
    s = "A" * (MAX_OUTPUT_BYTES + 100)
    out = _truncate_str(s)
    assert len(out) == MAX_OUTPUT_BYTES
    assert "output truncated" in out


def test_cap_output_trims_stream_field() -> None:
    out = nbformat.v4.new_output(
        output_type="stream", name="stdout", text="X" * (MAX_OUTPUT_BYTES + 5)
    )
    trimmed = _cap_output(out)
    assert trimmed is True
    assert len(out["text"]) == MAX_OUTPUT_BYTES


def test_cap_output_leaves_short_stream_alone() -> None:
    out = nbformat.v4.new_output(output_type="stream", name="stdout", text="short")
    assert _cap_output(out) is False
    assert out["text"] == "short"


def test_cap_output_does_not_touch_images() -> None:
    out = nbformat.v4.new_output(
        output_type="display_data",
        data={"image/png": "Z" * (MAX_OUTPUT_BYTES + 10)},
        metadata={},
    )
    trimmed = _cap_output(out)
    assert trimmed is False
    assert len(out["data"]["image/png"]) > MAX_OUTPUT_BYTES


def test_coalesce_same_stream_merges_text() -> None:
    out1 = nbformat.v4.new_output(output_type="stream", name="stdout", text="hello ")
    out2 = nbformat.v4.new_output(output_type="stream", name="stdout", text="world")
    outputs = [out1]
    coalesced = _coalesce_stream(outputs, out2)
    assert coalesced is True
    assert outputs[0]["text"] == "hello world"


def test_coalesce_different_streams_does_not_merge() -> None:
    out1 = nbformat.v4.new_output(output_type="stream", name="stdout", text="hi")
    out2 = nbformat.v4.new_output(output_type="stream", name="stderr", text="oops")
    outputs = [out1]
    assert _coalesce_stream(outputs, out2) is False


def test_output_size_counts_stream() -> None:
    out = nbformat.v4.new_output(output_type="stream", name="stdout", text="abcdef")
    assert _output_size(out) == 6


def test_output_size_skips_image_mime() -> None:
    """README/docs say image MIME doesn't count toward the cell cap."""
    out = nbformat.v4.new_output(
        output_type="display_data",
        data={
            "image/png": "Z" * (MAX_OUTPUT_BYTES + 10),
            "text/plain": "<Figure>",
        },
        metadata={},
    )
    # Only text/plain counts: image/png should be ignored.
    assert _output_size(out) == len("<Figure>")


def test_output_size_skips_widget_mime() -> None:
    out = nbformat.v4.new_output(
        output_type="display_data",
        data={
            "application/vnd.jupyter.widget-view+json": "X" * 100_000,
            "text/plain": "abc",
        },
        metadata={},
    )
    assert _output_size(out) == 3


def test_coalesce_stream_caps_merged_text() -> None:
    """Many small stream chunks must not accumulate past MAX_OUTPUT_BYTES."""
    prev = nbformat.v4.new_output(
        output_type="stream", name="stdout", text="A" * (MAX_OUTPUT_BYTES - 5)
    )
    chunk = nbformat.v4.new_output(
        output_type="stream", name="stdout", text="B" * 1000
    )
    outputs = [prev]
    coalesced = _coalesce_stream(outputs, chunk)
    assert coalesced is True
    assert len(outputs[0]["text"]) == MAX_OUTPUT_BYTES
    assert outputs[0].get("_mcp_truncated") is True
