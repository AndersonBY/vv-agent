"""Typed interaction helpers for producer tests."""

from vv_agent import AppServerUserInteraction


def target_of(form):
    return AppServerUserInteraction.from_dict(form).target


def displayed_target(driver, sid):
    waits = sorted(
        (w["interaction"] for _, _, w in driver.waits(sid) if "interaction" in w),
        key=lambda f: (f["sessionId"], f["operationId"] or ""),
    )
    return target_of(waits[0])


def text_action(form, text):
    return {"kind": "respond", "target": target_of(form).to_dict(), "answer": {"kind": "text", "text": text}}
