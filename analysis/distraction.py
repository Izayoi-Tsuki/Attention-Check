"""分心事件：从"专注"转入分心类状态的时刻。统计触发源与回到专注所需的时间。

每个窗口的状态代表时间段 [end - step, end)，相邻窗口拼成连续的时间线。
由于窗口有 30s 长度，状态切换会比实际行为滞后十几秒；回归耗时是两次切换之差，滞后大致抵消。
"""
from collections import defaultdict

import numpy as np

from .rules import DISTRACTED, runs, state_name

KIND_TRIGGER = {"LOOK_AWAY": "视线离开屏幕", "ZONED_OUT": "发呆（无操作、视线停滞）", "DROWSY": "犯困"}
MAX_NEUTRAL_GAP_S = 60  # 专注 → 一般（≤60s）→ 分心，仍算作一次从专注出发的分心
REGIONS = [["左上", "上方", "右上"], ["左侧", "中央", "右侧"], ["左下", "下方", "右下"]]


def episodes(F: dict, states: np.ndarray, step: float) -> list:
    ends = F["end"]
    return [dict(state=s, i0=i0, i1=i1, start=ends[i0] - step, end=ends[i1 - 1])
            for s, i0, i1 in runs(states)]


def gaze_region(x: float, y: float, screen) -> str:
    sw, sh = screen
    if not (0 <= x < sw and 0 <= y < sh):
        return "屏幕外"
    return REGIONS[min(int(y / sh * 3), 2)][min(int(x / sw * 3), 2)]


def _onset_off_task(d, ep_start: float, window_s: float):
    """任务外应用事件：在状态切换前一个窗口长度内，找第一次切到任务外窗口的那一秒。"""
    c = d.ctx
    a0 = np.searchsorted(c["ts"], ep_start - window_s)
    a1 = np.searchsorted(c["ts"], ep_start + 5)
    for i in range(a0, a1):
        if c["on_task"][i] == 0 and (i + 1 >= len(c["ts"]) or c["on_task"][i + 1] == 0):
            return c["ts"][i], c["label"][i], c["category"][i]
    i = min(a1, len(c["ts"]) - 1)
    return ep_start, c["label"][i], c["category"][i]


def find_events(d, F: dict, states: np.ndarray, cfg: dict) -> list:
    step, window_s = float(cfg["analysis"]["step_s"]), float(cfg["analysis"]["window_s"])
    cap = cfg["rules"]["return_cap_min"] * 60
    eps = episodes(F, states, step)
    events = []
    last_focus_end = None
    for k, ep in enumerate(eps):
        s = ep["state"]
        if s == "FOCUSED":
            last_focus_end = ep["end"]
            continue
        if s not in DISTRACTED:
            if s != "NEUTRAL":
                last_focus_end = None
            continue
        if last_focus_end is None or ep["start"] - last_focus_end > MAX_NEUTRAL_GAP_S:
            last_focus_end = None
            continue
        last_focus_end = None

        if s == "OFF_TASK_APP":
            onset, trigger, cat = _onset_off_task(d, ep["start"], window_s)
        else:
            onset, trigger, cat = ep["start"], KIND_TRIGGER[s], ""

        # 回到专注：下一个专注片段的开始；中途离开或停止记录则不计
        ret = None
        for nxt in eps[k + 1:]:
            if nxt["state"] in ("AWAY", "NO_DATA"):
                break
            if nxt["state"] == "FOCUSED":
                if nxt["start"] - ep["start"] <= cap:
                    ret = nxt["start"] - ep["start"]
                break

        region = ""
        if d.has_gaze:
            g = d.gaze
            g0, g1 = np.searchsorted(g["ts"], onset - 3), np.searchsorted(g["ts"], onset)
            x, y = g["x"][g0:g1], g["y"][g0:g1]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 3:
                region = gaze_region(float(np.median(x[ok])), float(np.median(y[ok])), d.screen)

        events.append(dict(ts=onset, kind=s, kind_name=state_name(s), trigger=trigger, category=cat,
                           duration_s=ep["end"] - ep["start"], return_s=ret, gaze_region=region))
    return events


def rank_sources(events: list) -> list:
    """按触发源聚合，按 次数 × 平均回归耗时 排序（回归耗时未知的事件按 1 分钟计入排序分）。"""
    groups = defaultdict(list)
    for e in events:
        groups[(e["kind_name"], e["trigger"], e["category"])].append(e)
    out = []
    for (kind, trigger, cat), es in groups.items():
        rets = [e["return_s"] for e in es if e["return_s"] is not None]
        mean_ret = float(np.mean(rets)) / 60 if rets else None
        out.append(dict(kind=kind, trigger=trigger, category=cat, count=len(es),
                        returned=len(rets), mean_return_min=mean_ret,
                        total_return_min=sum(rets) / 60,
                        score=len(es) * (mean_ret if mean_ret is not None else 1.0)))
    out.sort(key=lambda r: -r["score"])
    return out


def focus_streaks(eps: list) -> list:
    """每段连续专注的时长（秒）。"""
    return [e["end"] - e["start"] for e in eps if e["state"] == "FOCUSED"]
