"""Ambient-context channel: plugin background snapshots → proactive prompt.

The contract is ``push_message(..., ai_behavior="blind", visibility=[],
metadata={"context_type": "ambient", "ambient": {...}})``. ``blind`` is the
compatibility cornerstone, not an implementation detail: an older host maps it
to ``delivery_mode="silent"`` and ignores the event, so the same plugin is safe
against both. A newer host takes the snapshot out of the event flow before any
callback is built, and the snapshot then only ever reaches the prompt through
``get_ambient_contexts()`` — never the conversation queue, never a turn.

Two facts are load-bearing and asserted repeatedly below:

* ambient events must NOT reach ``enqueue_agent_callback`` /
  ``submit_proactive_callback`` (that is what ``read`` would do),
* everything that is not ambient must behave exactly as before.
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from config.prompts.prompts_proactive import (  # noqa: E402
    get_proactive_ambient_context_hint,
)
from main_logic.core import LLMSessionManager  # noqa: E402


def _make_mgr() -> LLMSessionManager:
    """Minimal real manager: only the attributes the ambient store touches.

    ``__init__`` needs a full config_manager, so we skip it with ``__new__`` —
    the store is created lazily, which is part of what this file pins down.
    """
    mgr = LLMSessionManager.__new__(LLMSessionManager)
    mgr.lanlan_name = "Test"
    return mgr


def _fake_mgr():
    """MagicMock stand-in for the main_server dispatch tests."""
    fake_mgr = MagicMock()
    fake_mgr.render_chat_blocks = AsyncMock(return_value=True)
    fake_mgr.enqueue_agent_callback = MagicMock()
    fake_mgr.submit_proactive_callback = MagicMock()
    fake_mgr.trigger_agent_callbacks = AsyncMock()
    fake_mgr.websocket = None
    fake_mgr._pending_agent_callback_task = None
    return fake_mgr


def _patch_main_server(monkeypatch, fake_mgr):
    from app import main_server  # noqa: F401  (imported by callers)

    monkeypatch.setattr(
        "app.main_server.character_runtime._get_session_manager",
        lambda name: fake_mgr,
    )
    monkeypatch.setattr(
        "app.main_server.character_runtime._is_websocket_connected",
        lambda ws: False,
    )


def _ambient_event(*, provider="now_playing", text="播放中：《晴天》— 周杰伦", **ambient):
    payload = {"provider": provider, "text": text}
    payload.update(ambient)
    return {
        "event_type": "proactive_message",
        "lanlan_name": "Test",
        "text": text,
        "channel": "plugin:now_playing",
        "task_id": "task-ambient",
        "delivery_mode": "silent",
        "ai_behavior": "blind",
        "visibility": [],
        "source_kind": "plugin",
        "source_name": "now_playing",
        "media_parts": [],
        "metadata": {"context_type": "ambient", "ambient": payload},
    }


# ──────────────────────────────────────────────────────────────────────
# Host dispatch: ambient events are stored, never queued
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
async def test_ambient_push_records_snapshot_and_never_enqueues(monkeypatch):
    """ambient + blind → note_ambient_context once; no callback, no submit."""
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    await main_server._handle_agent_event(_ambient_event(ttl_s=30, priority=5))

    fake_mgr.note_ambient_context.assert_called_once()
    kwargs = fake_mgr.note_ambient_context.call_args.kwargs
    assert kwargs["provider"] == "now_playing"
    assert kwargs["text"] == "播放中：《晴天》— 周杰伦"
    assert kwargs["ttl_s"] == 30
    assert kwargs["priority"] == 5
    # The channel this design exists to replace: read would enqueue, respond
    # would submit. Ambient does neither.
    fake_mgr.enqueue_agent_callback.assert_not_called()
    fake_mgr.submit_proactive_callback.assert_not_called()
    fake_mgr.trigger_agent_callbacks.assert_not_awaited()


@pytest.mark.unit
async def test_ambient_push_does_not_render_to_frontend(monkeypatch):
    """visibility=[] means no chat render either — it is not a conversation event."""
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    await main_server._handle_agent_event(_ambient_event())

    fake_mgr.render_chat_blocks.assert_not_awaited()


@pytest.mark.unit
async def test_ambient_text_falls_back_to_event_body(monkeypatch):
    """No ``ambient.text`` → the event's own summary/detail/text is used."""
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event()
    event["metadata"]["ambient"].pop("text")
    event["summary"] = "summary body"
    event["text"] = "raw body"

    await main_server._handle_agent_event(event)

    assert fake_mgr.note_ambient_context.call_args.kwargs["text"] == "summary body"


@pytest.mark.unit
async def test_non_ambient_blind_push_still_silent(monkeypatch):
    """REGRESSION: a plain blind push keeps its old behaviour exactly —
    silent, no queue, no turn. The ambient branch must not swallow it."""
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event()
    event["metadata"] = {}

    await main_server._handle_agent_event(event)

    fake_mgr.note_ambient_context.assert_not_called()
    fake_mgr.enqueue_agent_callback.assert_not_called()
    fake_mgr.submit_proactive_callback.assert_not_called()
    fake_mgr.trigger_agent_callbacks.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.parametrize(
    "ambient_payload",
    [
        ["not", "a", "dict"],
        {"text": "no provider at all"},
        {"provider": "BAD Name!", "text": "invalid provider"},
        {"provider": "", "text": "empty provider"},
        {"provider": 42, "text": "provider is not a string"},
    ],
)
async def test_malformed_ambient_metadata_is_dropped(monkeypatch, ambient_payload):
    """Malformed ambient metadata must not raise, must not fall through into
    the callback path (where it would surface as a chat event), and must not
    enqueue anything."""
    from app import main_server

    fake_mgr = _fake_mgr()
    # The real manager rejects these; the mock must too, or this test would
    # only prove that the mock was forgiving.
    fake_mgr.note_ambient_context.side_effect = (
        lambda **kwargs: isinstance(kwargs.get("provider"), str)
        and kwargs["provider"].isidentifier()
    )
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event()
    event["metadata"]["ambient"] = ambient_payload

    await main_server._handle_agent_event(event)  # must not raise

    fake_mgr.enqueue_agent_callback.assert_not_called()
    fake_mgr.submit_proactive_callback.assert_not_called()
    fake_mgr.render_chat_blocks.assert_not_awaited()


# ──────────────────────────────────────────────────────────────────────
# Manager store: replacement, ordering, expiry, sanitization, caps
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_same_provider_replaces_previous_snapshot():
    mgr = _make_mgr()

    assert mgr.note_ambient_context("now_playing", text="first") is True
    assert mgr.note_ambient_context("now_playing", text="second") is True

    entries = mgr.get_ambient_contexts()
    assert len(entries) == 1
    assert entries[0]["text"] == "second"


@pytest.mark.unit
def test_multiple_providers_sorted_by_priority_then_name():
    mgr = _make_mgr()
    mgr.note_ambient_context("weather", text="rain", priority=-5)
    mgr.note_ambient_context("game", text="act 2", priority=10)
    mgr.note_ambient_context("now_playing", text="song", priority=0)

    providers = [entry["provider"] for entry in mgr.get_ambient_contexts()]

    assert providers == ["game", "now_playing", "weather"]


@pytest.mark.unit
def test_expired_snapshot_disappears():
    mgr = _make_mgr()
    # ttl is clamped into [1, 3600], so the smallest legal TTL still takes a
    # second to expire. max_age_s covers the same filter instantly below.
    assert mgr.note_ambient_context("now_playing", text="song", ttl_s=0.01) is True
    assert len(mgr.get_ambient_contexts()) == 1

    time.sleep(1.05)

    assert mgr.get_ambient_contexts() == []


@pytest.mark.unit
def test_max_age_s_narrows_the_window_without_waiting():
    mgr = _make_mgr()
    mgr.note_ambient_context("now_playing", text="song", ttl_s=60)

    assert mgr.get_ambient_contexts(max_age_s=0) == []
    assert len(mgr.get_ambient_contexts(max_age_s=60)) == 1


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["BAD Name!", "", None, 42, "a" * 33])
def test_invalid_provider_is_rejected_without_raising(provider):
    mgr = _make_mgr()

    assert mgr.note_ambient_context(provider, text="whatever") is False
    assert mgr.get_ambient_contexts() == []


@pytest.mark.unit
def test_long_text_is_truncated():
    mgr = _make_mgr()

    assert mgr.note_ambient_context("now_playing", text="x" * 2000) is True

    assert len(mgr.get_ambient_contexts()[0]["text"]) <= 500


@pytest.mark.unit
def test_provider_cap_keeps_only_the_newest():
    mgr = _make_mgr()

    for index in range(20):
        assert mgr.note_ambient_context(f"provider_{index}", text=f"text {index}") is True

    entries = mgr.get_ambient_contexts()
    providers = {entry["provider"] for entry in entries}

    assert len(entries) == 16
    # The ones that just arrived survive; the earliest are the ones evicted.
    assert "provider_19" in providers
    assert "provider_0" not in providers


@pytest.mark.unit
def test_returned_entries_are_copies():
    mgr = _make_mgr()
    mgr.note_ambient_context("now_playing", text="song", data={"title": "晴天"})

    entries = mgr.get_ambient_contexts()
    entries[0]["text"] = "mutated"
    entries[0]["data"]["title"] = "mutated"

    fresh = mgr.get_ambient_contexts()
    assert fresh[0]["text"] == "song"
    assert fresh[0]["data"] == {"title": "晴天"}


@pytest.mark.unit
def test_non_dict_data_is_dropped_not_stored():
    mgr = _make_mgr()

    assert mgr.note_ambient_context("now_playing", text="song", data=["nope"]) is True

    assert mgr.get_ambient_contexts()[0]["data"] == {}


@pytest.mark.unit
def test_manager_without_store_reads_as_empty():
    """A ``__new__``-built manager that never wrote anything must not raise."""
    mgr = _make_mgr()

    assert mgr.get_ambient_contexts() == []


# ──────────────────────────────────────────────────────────────────────
# Rendering: multi-language, brace escaping, budget
# ──────────────────────────────────────────────────────────────────────


def _entry(text: str, provider: str = "now_playing", priority: int = 0) -> dict:
    return {
        "provider": provider,
        "text": text,
        "data": {},
        "ts": 0.0,
        "ttl_s": 30.0,
        "priority": priority,
    }


@pytest.mark.unit
def test_renderer_includes_fresh_text():
    rendered = get_proactive_ambient_context_hint(
        [_entry("正在播放《晴天》— 周杰伦")], None, "zh"
    )

    assert "正在播放《晴天》— 周杰伦" in rendered
    assert rendered.startswith("环境已知信息")


@pytest.mark.unit
def test_renderer_empty_input_is_empty_string():
    assert get_proactive_ambient_context_hint([], None, "zh") == ""
    assert get_proactive_ambient_context_hint([_entry("   ")], None, "zh") == ""
    assert get_proactive_ambient_context_hint(None, None, "zh") == ""


@pytest.mark.unit
def test_renderer_escapes_braces_so_outer_format_survives():
    """The hint is spliced into a prompt that is later .format()ed; a bracketed
    song title must survive that pass verbatim instead of raising KeyError."""
    rendered = get_proactive_ambient_context_hint([_entry("歌曲名是 {title} 与 {0}")], None, "zh")

    formatted = rendered.format()  # would raise KeyError/IndexError if unescaped
    assert "{title}" in formatted
    assert "{0}" in formatted


@pytest.mark.unit
def test_renderer_falls_back_to_english():
    zh = get_proactive_ambient_context_hint([_entry("x")], None, "zh")
    en = get_proactive_ambient_context_hint([_entry("x")], None, "en")
    unknown = get_proactive_ambient_context_hint([_entry("x")], None, "xx-YY")

    assert zh != en
    assert unknown == en


@pytest.mark.unit
def test_renderer_speaks_japanese_for_ja_locale():
    """Real Japanese wording, not merely "renders something non-empty".

    ``ja-JP`` is the tag the runtime locale set actually carries, so the
    normalizer path counts as part of the contract; a header reached through it
    must be the same row as the short ``ja`` key.
    """
    for lang in ("ja", "ja-JP"):
        rendered = get_proactive_ambient_context_hint(
            [_entry("「晴天」を再生中")], None, lang
        )

        assert rendered.startswith("環境の既知情報"), rendered
        assert "背景の事実" in rendered
        assert "「晴天」を再生中" in rendered
        # Not a fallback row: the neighbours it could fall through to must be absent.
        assert "Known ambient context" not in rendered
        assert "环境已知信息" not in rendered


@pytest.mark.unit
def test_header_covers_every_locale_the_sibling_music_hint_covers():
    """Key-set parity with ``PROACTIVE_MUSIC_PLAYING_HINT`` (the module's own
    reference table, backfilled for all eight locales). A missing row is silent
    today — the getter falls through to English — which is exactly why it needs
    a guard rather than a review checklist."""
    from config.prompts.prompts_proactive import (
        PROACTIVE_AMBIENT_CONTEXT_HEADER,
        PROACTIVE_MUSIC_PLAYING_HINT,
    )

    missing = set(PROACTIVE_MUSIC_PLAYING_HINT) - set(PROACTIVE_AMBIENT_CONTEXT_HEADER)
    assert missing == set(), f"这些 locale 会静默落回英文: {sorted(missing)}"


@pytest.mark.unit
def test_renderer_stops_at_the_character_budget():
    entries = [_entry("长" * 500, provider=f"provider_{index}") for index in range(6)]

    rendered = get_proactive_ambient_context_hint(entries, None, "zh")

    assert 0 < len(rendered) <= 1200


# ──────────────────────────────────────────────────────────────────────
# Wiring: hint builder → service slot
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_ambient_context_hint_reads_the_manager():
    from main_logic.proactive_chat.music_recommendation import (
        _build_ambient_context_hint,
    )

    mgr = _make_mgr()
    mgr.note_ambient_context("now_playing", text="正在播放《晴天》")

    hint = _build_ambient_context_hint(mgr=mgr, master_name="肥猫", lang="zh")

    assert "正在播放《晴天》" in hint


@pytest.mark.unit
def test_build_ambient_context_hint_tolerates_old_manager():
    """A manager without get_ambient_contexts (older host / partial double) → ""."""
    from main_logic.proactive_chat.music_recommendation import (
        _build_ambient_context_hint,
    )

    assert _build_ambient_context_hint(mgr=object(), master_name="", lang="zh") == ""

    exploding = MagicMock()
    exploding.get_ambient_contexts.side_effect = RuntimeError("boom")

    assert _build_ambient_context_hint(mgr=exploding, master_name="", lang="zh") == ""


@pytest.mark.unit
def test_no_ambient_leaves_the_slot_untouched():
    """Regression for the wiring: no snapshots → hint is "" → any caller that
    appends conditionally leaves the prompt byte-identical."""
    from main_logic.proactive_chat.music_recommendation import (
        _build_ambient_context_hint,
    )

    mgr = _make_mgr()
    music_playing_hint = "existing hint"

    ambient_hint = _build_ambient_context_hint(mgr=mgr, master_name="肥猫", lang="zh")
    merged = (
        f"{music_playing_hint}\n\n{ambient_hint}" if ambient_hint else music_playing_hint
    )

    assert ambient_hint == ""
    assert merged == music_playing_hint
