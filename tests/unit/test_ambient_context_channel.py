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
async def test_ambient_snapshot_without_any_event_body_still_records(monkeypatch):
    """A snapshot is metadata, not a message.

    The ambient branch used to sit behind ``if text or deferred_callback_images:``
    — i.e. behind the event BODY gate — so a metadata-only push (the normal case
    for a background fact, and the only case once a plugin stops bothering to
    duplicate its text into ``parts``) was dropped before it ever reached
    ``note_ambient_context``.
    """
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event(text="")
    event["metadata"]["ambient"].pop("text")
    for key in ("summary", "detail", "text"):
        event.pop(key, None)

    await main_server._handle_agent_event(event)

    fake_mgr.note_ambient_context.assert_called_once()
    assert fake_mgr.note_ambient_context.call_args.kwargs["text"] == ""
    fake_mgr.enqueue_agent_callback.assert_not_called()
    fake_mgr.render_chat_blocks.assert_not_awaited()


@pytest.mark.unit
async def test_ambient_with_chat_visibility_still_never_renders(monkeypatch):
    """A snapshot must not paint a bubble even if the plugin asks.

    The branch used to run AFTER the chat/HUD render, so ``visibility=["chat"]``
    on an ambient push produced a bubble — and the render is what makes a stale
    snapshot _durable_ in the transcript. Being handled before the whole event
    body is why this cannot regress: the render is unreachable from here.
    ``ai_behavior`` is deliberately wrong (``respond``) to prove metadata wins.
    """
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event()
    event["visibility"] = ["chat"]
    event["ai_behavior"] = "respond"

    await main_server._handle_agent_event(event)

    fake_mgr.note_ambient_context.assert_called_once()
    fake_mgr.render_chat_blocks.assert_not_awaited()
    fake_mgr.enqueue_agent_callback.assert_not_called()
    fake_mgr.submit_proactive_callback.assert_not_called()


@pytest.mark.unit
async def test_ambient_push_never_resolves_media_parts(monkeypatch):
    """No snapshot is worth an image fetch: handled before the parts block."""
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)
    resolve = MagicMock()
    monkeypatch.setattr(
        "app.main_server.character_runtime._resolve_plugin_model_image", resolve
    )

    event = _ambient_event()
    event["ai_behavior"] = "respond"
    event["media_parts"] = [{"type": "image", "data": "iVBORw0KGgo="}]

    await main_server._handle_agent_event(event)

    resolve.assert_not_called()
    fake_mgr.note_ambient_context.assert_called_once()


@pytest.mark.unit
async def test_ambient_omitted_ttl_defers_to_the_manager(monkeypatch):
    """No ``ttl_s`` in the payload → ``None``, not a second hard-coded 30.

    Two constants (host bridge + manager) can drift; the manager already owns a
    clamp and a default, so the bridge passes "no opinion" through.
    """
    from app import main_server

    fake_mgr = _fake_mgr()
    _patch_main_server(monkeypatch, fake_mgr)

    event = _ambient_event()
    event["metadata"]["ambient"].pop("ttl_s", None)

    await main_server._handle_agent_event(event)

    assert fake_mgr.note_ambient_context.call_args.kwargs["ttl_s"] is None


@pytest.mark.unit
async def test_non_ambient_blind_push_still_silent(monkeypatch):
    """A plain blind push keeps its old behaviour exactly —
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
def test_omitted_ttl_stores_the_host_default_ttl():
    """``ttl_s=None`` from the event bridge lands as the host default.

    Note what this does and does not pin: it locks the RESULT (30 s), not the
    ``if ttl_s is None`` branch — letting ``float(None)`` raise into the
    ``except`` yields the same 30 s. The branch buys readability, not
    behaviour; behaviour is covered by the boundary table below.
    """
    from main_logic.core.proactive import _AMBIENT_TTL_DEFAULT_S

    mgr = _make_mgr()
    assert mgr.note_ambient_context("now_playing", text="song", ttl_s=None) is True

    entry = mgr.get_ambient_contexts()[0]
    assert entry["ttl_s"] == _AMBIENT_TTL_DEFAULT_S
    assert 0 <= time.monotonic() - entry["ts"] < 5

    # And a snapshot with the default TTL is readable at once — a None
    # mishandled as 0 would already be expired by the line above.
    assert len(mgr.get_ambient_contexts()) == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "ttl_s, expected",
    [
        (None, 30.0),  # omitted → host default
        (0, 1.0),  # clamped up to the 1 s floor
        (0.01, 1.0),  # same floor, via a float
        (-5, 1.0),
        (10, 10.0),
        ("30", 30.0),  # numeric string survives float()
        (float("nan"), 30.0),  # NaN fails every comparison → default
        (3601, 3600.0),  # clamped down to the ceiling
        # bool is an int subclass, so True is 1 s (the floor). Nonsense input
        # (`ttl_s` is documented as a number) — pinned so it stays a floor
        # rather than silently becoming the 30 s default later.
        (True, 1.0),
        ("not-a-number", 30.0),  # unparseable → default, never raises
    ],
)
def test_ttl_boundaries_clamp_or_fall_back_to_the_default(ttl_s, expected):
    """Boundary table for the TTL contract: every endpoint and every malformed
    value, asserted on the stored snapshot (the value she actually gets).

    Written as a table on purpose — these hold regardless of whether the
    implementation clamps with ``min/max``, ``if``s, or lets ``float()`` raise.
    """
    mgr = _make_mgr()

    assert mgr.note_ambient_context("now_playing", text="song", ttl_s=ttl_s) is True

    entries = mgr.get_ambient_contexts()
    assert len(entries) == 1, f"ttl_s={ttl_s!r} should still be fresh"
    assert entries[0]["ttl_s"] == expected


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
    today — the getter falls through to English — which is exactly why the key
    set needs a guard rather than a convention."""
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

    hint = _build_ambient_context_hint(mgr=mgr, master_name="博士", lang="zh")

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
def test_phase2_tail_hints_keep_ambient_out_of_the_music_hint():
    """The seam a prompt-level assertion cannot reach.

    Concatenating the ambient block onto ``music_playing_hint`` renders the
    prompt byte-for-byte identically, so every render-level test stays green
    while the invariant "the music hint means the music hint" quietly dies.
    Asserting the SPLIT is the only way to hold it.
    """
    from main_logic.proactive_chat.music_recommendation import (
        _build_phase2_tail_hints,
    )

    mgr = _make_mgr()
    mgr.note_ambient_context("now_playing", text="正在播放《晴天》")

    music, extra = _build_phase2_tail_hints(
        mgr=mgr,
        is_playing_music=True,
        current_track={"name": "夜曲"},
        master_name="博士",
        lang="zh",
    )

    assert "夜曲" in music
    assert "晴天" not in music, "ambient must not ride the playback slot"
    assert len(extra) == 1
    assert "晴天" in extra[0]


@pytest.mark.unit
def test_phase2_tail_hints_are_empty_when_neither_source_has_anything():
    from main_logic.proactive_chat.music_recommendation import (
        _build_phase2_tail_hints,
    )

    music, extra = _build_phase2_tail_hints(
        mgr=_make_mgr(),
        is_playing_music=False,
        current_track=None,
        master_name="博士",
        lang="zh",
    )

    assert music == ""
    assert extra == ()


def _phase2_ctx(**overrides):
    """Phase2PromptContext with only the tail slot under test varying."""
    from main_logic.proactive_chat.generation import Phase2PromptContext

    base = dict(
        music_playing_hint="",
        character_prompt="CHAR",
        inner_thoughts="THOUGHTS",
        state_section="STATE",
        memory_context="MEM",
        recent_chats_section="CHATS",
        screen_section="",
        external_section="",
        music_section="",
        meme_section="",
        source_instruction="SRC",
        output_format_section="FMT",
    )
    base.update(overrides)
    return Phase2PromptContext(**base)


@pytest.mark.unit
def test_no_ambient_leaves_the_slot_untouched():
    """Regression for the wiring: no snapshots → hint is "" → the rendered
    Phase 2 prompt is byte-identical to one that never knew about ambient."""
    from main_logic.proactive_chat.music_recommendation import (
        _build_ambient_context_hint,
    )

    mgr = _make_mgr()
    ambient_hint = _build_ambient_context_hint(mgr=mgr, master_name="博士", lang="zh")
    assert ambient_hint == ""

    baseline = _phase2_ctx(music_playing_hint="existing hint").render(
        proactive_lang="zh", master_name="博士"
    )
    with_empty_slots = _phase2_ctx(
        music_playing_hint="existing hint", extra_context_hints=()
    ).render(proactive_lang="zh", master_name="博士")

    assert with_empty_slots == baseline


@pytest.mark.unit
def test_extra_context_hints_render_after_the_music_hint():
    """The generic slots land where the ambient block used to be spliced by
    hand: right after the playback hint, in the same placeholder."""
    music = "单曲循环《晴天》"
    ambient = "背景事实：正在播放《晴天》— 周杰伦"

    prompt = _phase2_ctx(
        music_playing_hint=music, extra_context_hints=(ambient,)
    ).render(proactive_lang="zh", master_name="博士")

    assert f"{music}\n\n{ambient}" in prompt


@pytest.mark.unit
def test_extra_context_hints_keep_their_order():
    prompt = _phase2_ctx(
        music_playing_hint="M", extra_context_hints=("A1", "A2")
    ).render(proactive_lang="zh", master_name="博士")

    assert "M\n\nA1\n\nA2" in prompt


@pytest.mark.unit
def test_empty_extra_hint_never_renders_a_blank_separator():
    """``""`` in the slots must not leave a ``\\n\\n`` behind:
    ``_build_ambient_context_hint`` may hand back an empty string without the
    caller filtering it."""
    baseline = _phase2_ctx(music_playing_hint="M").render(
        proactive_lang="zh", master_name="博士"
    )
    with_blank = _phase2_ctx(
        music_playing_hint="M", extra_context_hints=("",)
    ).render(proactive_lang="zh", master_name="博士")

    assert with_blank == baseline


@pytest.mark.unit
def test_extra_hint_alone_fills_the_slot():
    """No playback hint → the ambient block is the slot, exactly as the music
    hint would have been (no leading separator either)."""
    alone = _phase2_ctx(extra_context_hints=("SNAPSHOT",)).render(
        proactive_lang="zh", master_name="博士"
    )
    as_music = _phase2_ctx(music_playing_hint="SNAPSHOT").render(
        proactive_lang="zh", master_name="博士"
    )

    assert "SNAPSHOT" in alone
    assert alone == as_music
