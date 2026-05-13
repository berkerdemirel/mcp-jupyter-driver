"""Unit tests for widget MIME detection and snapshot parsing."""

from __future__ import annotations

from mcp_jupyter_driver.widgets import (
    WIDGET_STATE_MIME,
    WIDGET_VIEW_MIME,
    install_widget_state,
    outputs_contain_widget,
    parse_snapshot_stdout,
)


def test_outputs_contain_widget_true() -> None:
    outs = [{"output_type": "display_data", "data": {WIDGET_VIEW_MIME: {"model_id": "abc"}}}]
    assert outputs_contain_widget(outs)


def test_outputs_contain_widget_false() -> None:
    outs = [{"output_type": "display_data", "data": {"text/plain": "x"}}]
    assert not outputs_contain_widget(outs)


def test_parse_snapshot_well_formed() -> None:
    sentinel = "\x1emcp-widget-state\x1e"
    payload = {"version_major": 2, "state": {"m1": {"x": 1}}}
    text = f"prefix{sentinel}{__import__('json').dumps(payload)}{sentinel}suffix"
    parsed = parse_snapshot_stdout(text)
    assert parsed == payload


def test_parse_snapshot_missing_sentinels() -> None:
    assert parse_snapshot_stdout("nothing here") is None


def test_parse_snapshot_kernel_error() -> None:
    sentinel = "\x1emcp-widget-state\x1e"
    text = sentinel + '{"_mcp_error": "Whoops"}' + sentinel
    assert parse_snapshot_stdout(text) is None


def test_install_widget_state_writes_canonical_key() -> None:
    meta: dict = {}
    install_widget_state(meta, {"version_major": 2, "state": {}})
    assert WIDGET_STATE_MIME in meta["widgets"]
