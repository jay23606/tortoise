"""#10749 — the MCP health tool's own worst case must fit the deadline it is served under.

THE DEFECT. ``tortoise_health`` runs a DEEP probe inline, with an allowance resolved from
``monitoring.probe_setup_timeout()`` (20.0 s default, ceiling 300.0), while
``_await_under_mcp_wait_bound`` serves EVERY tool under ``mcp_auth._TRANSPORT_WAIT_BOUND_S``
(10.0 s) — no allowlist, no exemption, and **no transport exemption either**: on stdio and on the
self-hosted apps no arrival stamp exists, and the seam still binds the FULL bound
(``remaining = float(_mcp_auth._TRANSPORT_WAIT_BOUND_S)``). The handler's bounded tail is

    probe_setup_timeout() + PROBE_TIMEOUT + GRAPH_SIZE_TIMEOUT + QUOTA_STATE_TIMEOUT
  = 20.0 + 1.5 + 1.5 + 3.0 = 26.0 s                                 (2.6x the bound)

so a REACHABLE store is refused with the transport's own "-32009 wait budget exceeded" text,
which reads as "the queue is busy" and says nothing about the tool's budget.

MEASURED (2026-11-13, deployed process, from a box with the credential — issue comment
6862096657): the tool answered **2 of 27** attempts; the single timed success took a **10.77 s**
round trip and spent **6.96 s** of it in the DB probe, i.e. ~3.8 s of the bound is gone before
the probe's first byte, so refusal is certain once ``db.latency_ms`` exceeds ~6.2 s. The store
sat at 6.89-6.96 s, which is exactly the "occasionally ok, mostly refused" outcome observed.

WHAT THIS FILE PINS. The fix is shape B of the issue (clamp at the seam — no surface change):
the allowance the handler SPENDS is ``min(probe_setup_timeout(), remaining - reserve)`` where
``remaining`` is the deadline ``_await_under_mcp_wait_bound`` itself bound into
``_WAIT_BOUND_DEADLINE`` (so it covers stdio and self-hosted HTTP, where there is no arrival
stamp), ``reserve`` is the other three bounded phases plus a delivery margin, and the clamp is
VISIBLE on the report — as a BOOLEAN discriminator, because the duration beside it can be a
falsy ``0.0`` — so a clamped (unobserved) probe cannot be misread as a broken store (#3683's
``ok=None`` / ``status="unknown"``).
"""
from __future__ import annotations

import time

import pytest

from tortoise import mcp_auth, mcp_server, monitoring, quota

_BOUND = float(mcp_auth._TRANSPORT_WAIT_BOUND_S)


def _reserve() -> float:
    """The three other bounded phases plus the delivery margin the clamp must leave behind."""
    return (monitoring.PROBE_TIMEOUT
            + monitoring.GRAPH_SIZE_TIMEOUT
            + quota.QUOTA_STATE_TIMEOUT
            + mcp_server._HEALTH_TAIL_MARGIN_S)


@pytest.fixture(autouse=True)
def _http_transport():
    """Drive the tool body directly, under the transport mode it actually runs in.

    ``_safe`` fail-closes on ``_transport_mode is None`` (the #236 gate), and the stdio arm
    further refuses once ``TORTOISE_API_KEY`` is present — which is true on every cell that
    carries the hosted credential. The ``http`` arm delegates auth to the transport, which is
    what a tool-body test wants. Same fixture as tests/test_10602_query_limit_bound.py.
    """
    from tortoise.mcp_auth import (  # noqa: I001
        _current_org_id, _current_org_limits, _transport_mode,
    )
    _transport_mode.set("http")
    _current_org_id.set(None)
    _current_org_limits.set(None)
    yield
    _transport_mode.set(None)
    _current_org_id.set(None)
    _current_org_limits.set(None)


@pytest.fixture(autouse=True)
def _no_seam_binding():
    """Start every test from "the seam never ran", so a test opts IN to a deadline explicitly.

    ``_WAIT_BOUND_DEADLINE`` is a module-level ContextVar; without this the value one test binds
    would leak into the next and make the unclamped-arm test pass for the wrong reason.
    """
    token = mcp_server._WAIT_BOUND_DEADLINE.set(None)
    yield
    mcp_server._WAIT_BOUND_DEADLINE.reset(token)


def _capture(monkeypatch):
    """Patch the report's collaborators and capture the allowance the handler actually spends."""
    seen: dict = {}

    def fake_metrics(**kwargs):
        seen.update(kwargs)
        return {"status": "unknown", "db": {"ok": None, "observed": False}}

    monkeypatch.setattr(monitoring, "metrics", fake_metrics)
    monkeypatch.setattr(mcp_server, "_org_quota_block", lambda sdk: {"state": "unknown"})
    monkeypatch.setattr(mcp_server, "_get_org_sdk", lambda: object())
    return seen


def test_the_unclamped_tail_outgrows_the_transport_bound():
    """The defect itself, in one line: 26.0 s of allowance inside a 10.0 s bound.

    This assertion is deliberately the OPPOSITE of what the fix guarantees — it documents why
    a clamp has to exist. If a future change brings the four constants under the bound on their
    own, this test reds, and that is the signal to delete the clamp rather than keep both.
    """
    tail = (monitoring.probe_setup_timeout()
            + monitoring.PROBE_TIMEOUT
            + monitoring.GRAPH_SIZE_TIMEOUT
            + quota.QUOTA_STATE_TIMEOUT)
    assert tail > mcp_auth._TRANSPORT_WAIT_BOUND_S, (
        f"tail {tail}s is no longer larger than the transport bound "
        f"{mcp_auth._TRANSPORT_WAIT_BOUND_S}s — the clamp may no longer be needed"
    )


def test_the_clamped_allowance_fits_the_remaining_deadline():
    """The invariant this issue asks for: health budget <= transport bound - margin."""
    bound = mcp_auth._TRANSPORT_WAIT_BOUND_S
    allowance = mcp_server._health_setup_allowance(bound)
    assert allowance == pytest.approx(bound - _reserve())
    assert allowance + _reserve() <= bound


def test_the_clamp_never_answers_a_negative_allowance():
    """A spent deadline yields 0.0, never a negative timeout (which would raise in the probe)."""
    assert mcp_server._health_setup_allowance(0.0) == 0.0
    assert mcp_server._health_setup_allowance(1.0) == 0.0


def test_a_generous_deadline_keeps_the_3143_allowance():
    """The clamp only ever REDUCES: #3143's cold-start allowance survives when it fits."""
    assert (mcp_server._health_setup_allowance(mcp_auth._TRANSPORT_WAIT_BOUND_S + 300.0)
            == monitoring.probe_setup_timeout())


def test_the_margin_is_a_real_reserve_not_zero():
    """A zero margin would make the issue's own target (`<= bound − margin`) vacuous.

    Every other assertion on the margin derives it from the same constant the implementation
    uses, so those move together: setting `_HEALTH_TAIL_MARGIN_S = 0.0` would leave them all
    green while the handler has NO room left for serialisation and delivery after its last
    bounded phase — which is the refusal this change exists to remove, back at the margin.
    """
    assert mcp_server._HEALTH_TAIL_MARGIN_S >= 0.25, (
        "the delivery margin was reduced to a value that leaves no room for the response "
        "itself; the clamp then re-introduces the -32009 refusal it was added to prevent"
    )
    phases = (monitoring.PROBE_TIMEOUT + monitoring.GRAPH_SIZE_TIMEOUT
              + quota.QUOTA_STATE_TIMEOUT)
    assert _reserve() >= phases + 0.25, (
        "the reserve no longer contains a delivery margin on top of the three bounded phases"
    )
    bound = mcp_auth._TRANSPORT_WAIT_BOUND_S
    allowance = mcp_server._health_setup_allowance(bound)
    assert allowance + _reserve() == pytest.approx(bound), (
        "the reserve is not fully accounted: part of the deadline is neither spent nor reserved"
    )


def test_the_reserve_leaves_room_inside_the_bound():
    """A static guard on the four constants: a future bump must not make the reserve unfittable.

    If ``PROBE_TIMEOUT + GRAPH_SIZE_TIMEOUT + QUOTA_STATE_TIMEOUT + margin`` ever reaches the
    transport bound, the clamp degenerates to "always 0" and the tool could never report a real
    probe again — silently. This reds first.
    """
    assert _reserve() < mcp_auth._TRANSPORT_WAIT_BOUND_S


def test_a_direct_in_process_call_keeps_the_unclamped_allowance(monkeypatch):
    """``None`` from the seam means the seam NEVER RAN (a direct call), not "no deadline".

    This is the one state where #3143's allowance is still the right answer: nothing is enforcing
    a transport bound, so there is nothing to clamp under. The two existing direct-call tests in
    ``tests/test_monitoring.py`` rely on exactly this.
    """
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    assert mcp_server._WAIT_BOUND_DEADLINE.get() is None
    seen = _capture(monkeypatch)

    mcp_server.tortoise_health()

    assert seen["setup_timeout"] == monitoring.probe_setup_timeout()


def test_stdio_the_seam_still_binds_the_full_transport_bound(monkeypatch):
    """RED on the pre-review head: the clamp was skipped whenever no arrival stamp existed.

    On stdio AND on the self-hosted apps no stamp is written (only ``hosted_api``'s middleware
    writes one), but ``_await_under_mcp_wait_bound`` still binds the FULL bound there. The pre-fix
    helper read the absent stamp as "no deadline" and spent the unclamped 20.0 s inside a 10.0 s
    bound — #10749's defect preserved on a first-class transport, with the test blessing it.
    """
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    token = mcp_server._WAIT_BOUND_DEADLINE.set(
        time.monotonic() + float(mcp_auth._TRANSPORT_WAIT_BOUND_S))
    try:
        seen = _capture(monkeypatch)
        report = mcp_server.tortoise_health()
    finally:
        mcp_server._WAIT_BOUND_DEADLINE.reset(token)

    bound = mcp_auth._TRANSPORT_WAIT_BOUND_S
    assert seen["setup_timeout"] <= bound - _reserve()
    assert seen["setup_timeout"] < monitoring.probe_setup_timeout()
    assert report.get("setup_timeout_clamped") is True


def test_the_handler_clamps_under_a_live_transport_deadline(monkeypatch):
    """On HTTP the allowance is clamped to the REMAINING deadline, not the full one."""
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: time.monotonic())
    seen = _capture(monkeypatch)

    mcp_server.tortoise_health()

    spent = seen["setup_timeout"]
    assert spent <= mcp_auth._TRANSPORT_WAIT_BOUND_S - _reserve()
    assert spent < monitoring.probe_setup_timeout()
    assert spent >= 0.0


def test_an_aged_arrival_stamp_reduces_the_allowance(monkeypatch):
    """The clamp must SUBTRACT elapsed request time — the measured case had 3.8 s already spent.

    A clamp that ignored elapsed time (``min(probe_setup_timeout(), bound - reserve)``) passes
    every other test in this file, because they all use ``time.monotonic()`` — the one arrival
    value where "spend what is left" and "spend the full bound" are the same number. At the
    deployment this issue measured, remaining was ~6.2 s, so an elapsed-ignoring clamp grants
    3.5 s, the tail becomes 10.0 s, and the -32009 refusal returns silently green.
    """
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: time.monotonic() - 3.0)
    seen = _capture(monkeypatch)

    mcp_server.tortoise_health()

    bound = mcp_auth._TRANSPORT_WAIT_BOUND_S
    assert seen["setup_timeout"] <= bound - 3.0 - _reserve()
    assert seen["setup_timeout"] != pytest.approx(bound - _reserve(), abs=0.10), (
        "the allowance ignored the ~3 s already spent on this request"
    )


def test_the_report_says_it_was_clamped_with_a_truthy_discriminator(monkeypatch):
    """A clamped probe reports ``unknown`` for a reason; the report must name that reason.

    Without the marker a clamped probe is indistinguishable from a broken store — #3143's
    symptom in a new dress, which is the one thing the issue says shape B must not reintroduce.
    The BOOLEAN is what a reader tests: the duration beside it is legitimately ``0.0`` on a spent
    deadline, and ``if report.get("setup_timeout_clamped_s")`` would read the most severe clamp
    as "not clamped".
    """
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: time.monotonic())
    _capture(monkeypatch)

    report = mcp_server.tortoise_health()

    assert report["setup_timeout_clamped"] is True
    assert report["setup_timeout_clamped_s"] <= mcp_auth._TRANSPORT_WAIT_BOUND_S - _reserve()
    assert report["setup_timeout_clamped_s"] < monitoring.probe_setup_timeout()


def test_a_spent_deadline_still_reads_as_clamped(monkeypatch):
    """The 0.0 case — the most severe clamp — must not read as "not clamped" (falsy duration)."""
    token = mcp_server._WAIT_BOUND_DEADLINE.set(time.monotonic())
    try:
        _capture(monkeypatch)
        report = mcp_server.tortoise_health()
    finally:
        mcp_server._WAIT_BOUND_DEADLINE.reset(token)

    assert report["setup_timeout_clamped_s"] == 0.0
    assert report["setup_timeout_clamped"] is True


def test_an_unclamped_report_carries_no_clamp_marker(monkeypatch):
    """The marker is a claim about THIS call: it must be absent when nothing was clamped."""
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    _capture(monkeypatch)

    report = mcp_server.tortoise_health()

    assert "setup_timeout_clamped" not in report
    assert "setup_timeout_clamped_s" not in report


def test_a_clamped_probe_is_unknown_not_degraded():
    """The clamp's safety claim, driven through the REAL ``monitoring.metrics``.

    Every handler-level test above replaces ``monitoring.metrics`` with a fake, so they would all
    stay green if ``metrics(setup_timeout=0.0)`` ever raised, blocked, or reported ``degraded`` —
    a false alarm on a HEALTHY store, which is the failure this whole change exists to avoid. This
    test pins the chain the clamp depends on: an allowance of 0.0 makes the probe UNOBSERVED, and
    ``metrics`` maps that to ``ok=None`` / ``status="unknown"`` (#3683), never ``degraded``.
    """
    class SlowColdStart:
        """A reachable store whose projection cold-start outlives any allowance given."""

        def _get_proj(self):
            time.sleep(0.4)
            raise AssertionError("the probe should have given up before this returned")

        def _cache_meta(self, *a, **k):
            return None

    started = time.monotonic()
    report = monitoring.metrics(sdk=SlowColdStart(), setup_timeout=0.0)
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"a 0.0 allowance still spent {elapsed:.2f}s (the bound is 10.0s)"
    assert report["status"] == "unknown"
    assert report["db"]["ok"] is None
    assert report["db"]["observed"] is False


@pytest.mark.asyncio
async def test_the_seam_binds_its_deadline_where_the_tool_body_can_read_it(monkeypatch):
    """The P1's actual mechanism: the number the SEAM enforces is the number the body reads.

    The clamp is only correct if it clamps under the deadline ``_await_under_mcp_wait_bound``
    itself computed. Re-deriving it in the body diverged on the no-arrival-stamp arm (stdio and
    the self-hosted apps), which is how the first version of this change left #10749 unfixed on
    those transports. This drives the REAL seam with a stand-in inner handler, so removing the
    binding — or resetting it before the task is created — reds here.
    """
    seen: dict = {}

    async def fake_inner(name, arguments, **kwargs):
        seen["deadline"] = mcp_server._WAIT_BOUND_DEADLINE.get()
        seen["remaining_after_bind"] = mcp_server._mcp_remaining_wait_bound()
        return {"ok": True}

    monkeypatch.setattr(mcp_server, "_original_call_tool", fake_inner)
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    monkeypatch.setattr(mcp_server, "_wait_bound_refused", lambda: False)
    monkeypatch.setattr(mcp_server, "_hold_mcp_dispatch_after_request", lambda *a, **k: None)

    await mcp_server._await_under_mcp_wait_bound(
        "tortoise_health", {}, version=None, task_meta=None)

    assert seen["deadline"] is not None, (
        "the seam did not bind a deadline for the tool body to clamp under — on stdio and "
        "self-hosted HTTP that leaves tortoise_health spending its unclamped 26.0 s tail"
    )
    assert seen["deadline"] == pytest.approx(time.monotonic() + _BOUND, abs=5.0), (
        "the seam bound something other than the absolute deadline it enforces"
    )
    assert seen["remaining_after_bind"] == pytest.approx(_BOUND, abs=1.0), (
        "the body did not read back approximately the full bound"
    )


def test_time_spent_after_the_seam_binds_reduces_the_allowance(monkeypatch):
    """A DEADLINE, not a duration — the fix must deduct whatever the dispatch already spent.

    The seam's own ``wait_for`` clock starts when it binds, so a duration captured there grants an
    allowance the handler no longer has. Measured on the cycle-2 head with a 4.0 s in-dispatch spend
    (the slot ``_resolve_graph_argument``'s control-plane lookup occupies): the duration shape
    granted the full ``bound - reserve`` and the worst case became ``4.0 + 3.5 + 1.5 + 1.5 + 3.0 =
    13.5 s`` inside a 10.0 s bound — the ``-32009`` refusal this change exists to remove, re-armed
    silently. This pins the deadline read at the point of USE.
    """
    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    token = mcp_server._WAIT_BOUND_DEADLINE.set(time.monotonic() + _BOUND)
    try:
        time.sleep(2.0)          # the dispatch spending time before the handler clamps
        seen = _capture(monkeypatch)
        mcp_server.tortoise_health()
    finally:
        mcp_server._WAIT_BOUND_DEADLINE.reset(token)

    assert seen["setup_timeout"] <= _BOUND - 2.0 - _reserve(), (
        "the allowance ignored the ~2 s spent after the seam bound its deadline; the handler "
        "then claims time it no longer has and the transport refuses the call"
    )


def test_a_clamp_raises_the_aggregate_counter_and_an_unclamped_call_does_not(monkeypatch):
    """The counter is the only aggregate that keeps "health is being clamped" answerable.

    The clamp moves a degradation from a transport REFUSAL (counted by
    ``wait_bound_breaches``) to an honest ``status="unknown"`` (counted by nothing else), so an
    unguarded counter is exactly the uncounted-symptom condition that finding was about.
    """
    baseline = monitoring.health_setup_clamped_count()

    monkeypatch.setattr(mcp_server, "_wait_bound_arrival", lambda: None)
    _capture(monkeypatch)
    mcp_server.tortoise_health()                       # unclamped arm
    assert monitoring.health_setup_clamped_count() == baseline, (
        "an UNCLAMPED health call incremented the clamp counter"
    )

    token = mcp_server._WAIT_BOUND_DEADLINE.set(time.monotonic() + _BOUND)
    try:
        _capture(monkeypatch)
        mcp_server.tortoise_health()                   # clamped arm
    finally:
        mcp_server._WAIT_BOUND_DEADLINE.reset(token)

    assert monitoring.health_setup_clamped_count() == baseline + 1, (
        "a clamped health call did not increment the clamp counter"
    )


def test_metrics_publishes_the_clamp_counter_as_an_int():
    """The key is what an operator reads on the standalone /health and /metrics surfaces.

    Without this assertion the key can be dropped from ``metrics()`` with the whole suite green —
    the per-call boolean is pinned by five tests, its aggregate twin by none.
    """
    report = monitoring.metrics()
    assert "health_setup_clamped" in report
    assert isinstance(report["health_setup_clamped"], int)


def test_the_hosted_health_body_publishes_the_clamp_counter():
    """The hosted app builds its OWN body — ``metrics()``'s key never reaches that surface.

    ``wait_bound_breaches`` is published there explicitly because it is "the surface an operator
    curls FIRST"; the clamp makes that counter go flat by design, so its positive pair must be
    published on the same body or the operator's first curl cannot see the degradation at all.

    ⚠️ Asserted on the BODY, not a source grep: ``inspect.getsource`` is satisfied by a mention (a
    comment, or the key nested under another block), so that version stayed green if the top-level
    key the runbook's ``jq '.health_setup_clamped'`` reads were renamed or moved to ``null``. Same
    shape as ``tests/test_10602_wait_bound_breach_visible.py``, which asserts on the served body.
    """
    import asyncio

    from tortoise import hosted_api

    body = asyncio.run(hosted_api.health())

    assert "health_setup_clamped" in body
    assert isinstance(body["health_setup_clamped"], int)
