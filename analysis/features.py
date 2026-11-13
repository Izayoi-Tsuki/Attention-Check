"""把原始数据切成滑动窗口（默认 30s 窗口、5s 步长），计算每个窗口的特征。

窗口以结束时刻 end 标识，覆盖 [end - window_s, end)。
没有视线数据（无摄像头 / 未校准）的窗口，视线特征为 NaN。
"""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from storage.db import connect

CTX_COLS = ["ts", "exe", "title", "category", "label", "on_task",
            "win_l", "win_t", "win_r", "win_b", "keys", "clicks", "mouse_moves", "scrolls", "idle_s"]
CTX_TEXT = {"exe", "title", "category", "label"}
GAZE_COLS = ["ts", "face", "valid", "x", "y", "blink_l", "blink_r", "yaw", "pitch"]


@dataclass
class DayData:
    t0: float
    t1: float
    screen: tuple
    ctx: dict = field(default_factory=dict)
    gaze: dict = field(default_factory=dict)
    probes: list = field(default_factory=list)     # (ts, session_id, source, answer, response_ms)
    sessions: list = field(default_factory=list)   # (id, start, end, task)
    events: list = field(default_factory=list)     # (ts, session_id, kind, detail)

    @property
    def has_gaze(self) -> bool:
        return len(self.gaze["ts"]) > 0 and bool(np.isfinite(self.gaze["x"]).any())


def load_range(db_path: str, t0: float, t1: float, screen=None) -> DayData:
    conn = connect(db_path)
    try:
        rows = conn.execute(f"SELECT {','.join(CTX_COLS)} FROM context WHERE ts>=? AND ts<? ORDER BY ts",
                            (t0, t1)).fetchall()
        ctx = {}
        for i, c in enumerate(CTX_COLS):
            col = [r[i] for r in rows]
            ctx[c] = (np.array([v or "" for v in col], dtype=object) if c in CTX_TEXT
                      else np.array(col, dtype=float).reshape(-1))
        rows = conn.execute(f"SELECT {','.join(GAZE_COLS)} FROM gaze WHERE ts>=? AND ts<? ORDER BY ts",
                            (t0, t1)).fetchall()
        arr = np.array(rows, dtype=float).reshape(-1, len(GAZE_COLS))  # NULL → NaN
        gaze = {c: arr[:, i] for i, c in enumerate(GAZE_COLS)}
        probes = conn.execute("SELECT ts, session_id, source, answer, response_ms FROM probes "
                              "WHERE ts>=? AND ts<? ORDER BY ts", (t0, t1)).fetchall()
        sessions = conn.execute("SELECT id, start, end, task FROM sessions WHERE start<? AND COALESCE(end, start)>=? "
                                "ORDER BY start", (t1, t0)).fetchall()
        events = conn.execute("SELECT ts, session_id, kind, detail FROM events WHERE ts>=? AND ts<? ORDER BY ts",
                              (t0, t1)).fetchall()
        if screen is None:
            r = conn.execute("SELECT screen_w, screen_h FROM calibrations ORDER BY id DESC LIMIT 1").fetchone()
            screen = tuple(r) if r else None
    finally:
        conn.close()
    if screen is None:
        try:
            from capture.screen import screen_size
            screen = screen_size()
        except Exception:
            screen = (1920, 1080)
    return DayData(t0, t1, screen, ctx, gaze, probes, sessions, events)


# ---------- 视线事件 ----------
def detect_fixations(ts, x, y, max_disp: float, min_dur: float, max_gap: float = 0.5):
    """I-DT 的增量版本：窗口离散度 (max-min)x + (max-min)y 超过阈值或出现断帧时结束当前注视。

    返回 (start, end, cx, cy) 四个数组。输入只应包含有效视线点。
    """
    n = len(ts)
    out = []
    if n == 0:
        return tuple(np.zeros(0) for _ in range(4))
    s = 0
    x0 = x1 = x[0]
    y0 = y1 = y[0]
    for j in range(1, n + 1):
        if j < n:
            nx0, nx1 = min(x0, x[j]), max(x1, x[j])
            ny0, ny1 = min(y0, y[j]), max(y1, y[j])
            if ts[j] - ts[j - 1] <= max_gap and (nx1 - nx0) + (ny1 - ny0) <= max_disp:
                x0, x1, y0, y1 = nx0, nx1, ny0, ny1
                continue
        if ts[j - 1] - ts[s] >= min_dur:
            out.append((ts[s], ts[j - 1], x[s:j].mean(), y[s:j].mean()))
        if j < n:
            s = j
            x0 = x1 = x[j]
            y0 = y1 = y[j]
    if not out:
        return tuple(np.zeros(0) for _ in range(4))
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def detect_blinks(ts, closed, max_gap: float = 0.5):
    """连续闭眼帧合并为一次眨眼，返回 (开始时刻, 持续秒数)。"""
    starts, durs = [], []
    i, n = 0, len(ts)
    while i < n:
        if not closed[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and closed[j + 1] and ts[j + 1] - ts[j] <= max_gap:
            j += 1
        end = ts[j + 1] if j + 1 < n and ts[j + 1] - ts[j] <= max_gap else ts[j]
        starts.append(ts[i])
        durs.append(min(end - ts[i], 3.0))  # 低头看键盘也会被识别为闭眼，截断避免拉高均值
        i = j + 1
    return np.array(starts, dtype=float), np.array(durs, dtype=float)


def _csum(a):
    return np.concatenate([[0.0], np.cumsum(np.asarray(a, dtype=float))])


def _wsum(c, i0, i1):
    return c[i1] - c[i0]


def _ratio(num, den, min_den=1):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= min_den, num / np.maximum(den, 1), np.nan)


# ---------- 窗口特征 ----------
def compute_windows(d: DayData, cfg: dict) -> dict:
    a = cfg["analysis"]
    W, S = float(a["window_s"]), float(a["step_s"])
    sample_s = float(cfg["context"]["sample_seconds"])
    c = d.ctx
    cts = c["ts"]
    if len(cts) == 0:
        return {"end": np.zeros(0)}

    ends = np.arange(np.floor(cts[0] / S) * S + S, cts[-1] + S + 1e-6, S)
    n = len(ends)
    i0 = np.searchsorted(cts, ends - W)
    i1 = np.searchsorted(cts, ends)
    n_ctx = (i1 - i0).astype(float)
    F = dict(end=ends, n_ctx=n_ctx, coverage=np.minimum(n_ctx * sample_s / W, 1.0),
             hour=np.array([(e - d.t0) / 3600 for e in ends]))

    # 上下文：键鼠频率（次/分钟）、任务外占比、应用切换次数
    span_min = np.maximum(n_ctx * sample_s, 1.0) / 60
    for k in ("keys", "clicks", "mouse_moves", "scrolls"):
        F[f"{k}_pm"] = _wsum(_csum(c[k]), i0, i1) / span_min
    F["off_task_ratio"] = _ratio(_wsum(_csum(1 - c["on_task"]), i0, i1), n_ctx)
    F["distraction_ratio"] = _ratio(_wsum(_csum(c["category"] == "distraction"), i0, i1), n_ctx)
    F["comms_ratio"] = _ratio(_wsum(_csum(c["category"] == "comms"), i0, i1), n_ctx)
    change = np.concatenate([[0], (c["label"][1:] != c["label"][:-1]) | (c["exe"][1:] != c["exe"][:-1])])
    cc = _csum(change)
    F["switches"] = cc[i1] - cc[np.minimum(i0 + 1, i1)]

    input_gap = np.full(n, np.nan)
    idle_end = np.full(n, np.nan)
    label_end = np.full(n, "", dtype=object)
    exe_end = np.full(n, "", dtype=object)
    cat_end = np.full(n, "", dtype=object)
    off_label = np.full(n, "", dtype=object)
    for k in range(n):
        a0, a1 = i0[k], i1[k]
        if a1 <= a0:
            continue
        # 窗口内最长无操作时长：idle_s 是距上次输入的秒数，截断到窗口起点，避免把窗口之前的空闲算进来
        input_gap[k] = np.max(np.minimum(c["idle_s"][a0:a1], cts[a0:a1] - (ends[k] - W)))
        idle_end[k] = c["idle_s"][a1 - 1]
        label_end[k], exe_end[k], cat_end[k] = c["label"][a1 - 1], c["exe"][a1 - 1], c["category"][a1 - 1]
        off = c["on_task"][a0:a1] == 0
        if off.any():
            off_label[k] = Counter(c["label"][a0:a1][off]).most_common(1)[0][0]
    F.update(input_gap=input_gap, idle_end=idle_end, label_end=label_end, exe_end=exe_end,
             category_end=cat_end, off_label=off_label)

    _gaze_features(d, cfg, F, ends, W)
    return F


def _gaze_features(d: DayData, cfg: dict, F: dict, ends, W):
    a = cfg["analysis"]
    g = d.gaze
    n = len(ends)
    gts = g["ts"]
    names = ["n_gaze", "face_ratio", "n_pos", "offscreen_ratio", "in_window_ratio", "dispersion",
             "fix_count", "fix_mean", "fix_max", "blink_rate", "blink_dur", "perclos", "yaw_std", "pitch_std"]
    for k in names:
        F[k] = np.full(n, np.nan)
    F["n_gaze"] = np.zeros(n)
    F["n_pos"] = np.zeros(n)
    if len(gts) == 0:
        return

    sw, sh = d.screen
    face = g["face"] == 1
    pos = (g["valid"] == 1) & np.isfinite(g["x"]) & np.isfinite(g["y"])
    mx, my = 0.05 * sw, 0.05 * sh
    on_screen = pos & (g["x"] >= -mx) & (g["x"] < sw + mx) & (g["y"] >= -my) & (g["y"] < sh + my)
    blink = np.fmax(g["blink_l"], g["blink_r"])
    blink = np.where(np.isfinite(blink), blink, 0)
    closed_perclos = face & ((np.nan_to_num(g["blink_l"]) + np.nan_to_num(g["blink_r"])) / 2 > a["blink_closed"])

    # 视线是否落在当时的前台窗口内
    c = d.ctx
    ci = np.clip(np.searchsorted(c["ts"], gts, side="right") - 1, 0, len(c["ts"]) - 1)
    fresh = np.abs(gts - c["ts"][ci]) <= 2 * cfg["context"]["sample_seconds"]
    in_win = on_screen & fresh & (g["x"] >= c["win_l"][ci]) & (g["x"] < c["win_r"][ci]) \
        & (g["y"] >= c["win_t"][ci]) & (g["y"] < c["win_b"][ci])

    g0 = np.searchsorted(gts, ends - W)
    g1 = np.searchsorted(gts, ends)
    n_gaze = (g1 - g0).astype(float)
    n_face = _wsum(_csum(face), g0, g1)
    n_pos = _wsum(_csum(pos), g0, g1)
    n_on = _wsum(_csum(on_screen), g0, g1)
    F["n_gaze"], F["n_pos"] = n_gaze, n_pos
    has = n_gaze >= max(10, W * 0.5)  # 至少约 0.5 fps
    F["face_ratio"] = np.where(has, _ratio(n_face, n_gaze), np.nan)
    pos_ok = has & (n_pos >= 10)
    F["offscreen_ratio"] = np.where(pos_ok, 1 - _ratio(n_on, n_pos), np.nan)
    F["in_window_ratio"] = np.where(pos_ok, _ratio(_wsum(_csum(in_win), g0, g1), n_pos), np.nan)
    F["perclos"] = np.where(has, _ratio(_wsum(_csum(closed_perclos), g0, g1), n_face, 5), np.nan)

    # 屏幕内视线的离散度：sqrt(var x + var y)
    xs = np.where(on_screen, g["x"], 0.0)
    ys = np.where(on_screen, g["y"], 0.0)
    sx, sxx = _wsum(_csum(xs), g0, g1), _wsum(_csum(xs * xs), g0, g1)
    sy, syy = _wsum(_csum(ys), g0, g1), _wsum(_csum(ys * ys), g0, g1)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (sxx / n_on - (sx / n_on) ** 2) + (syy / n_on - (sy / n_on) ** 2)
    F["dispersion"] = np.where(pos_ok & (n_on >= 10), np.sqrt(np.maximum(var, 0)), np.nan)

    yaw = np.where(face, np.nan_to_num(g["yaw"]), 0.0)
    pit = np.where(face, np.nan_to_num(g["pitch"]), 0.0)
    for name, v in (("yaw_std", yaw), ("pitch_std", pit)):
        s1, s2 = _wsum(_csum(v), g0, g1), _wsum(_csum(v * v), g0, g1)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 / n_face - (s1 / n_face) ** 2
        F[name] = np.where(has & (n_face >= 5), np.sqrt(np.maximum(var, 0)), np.nan)

    # 眨眼
    b_start, b_dur = detect_blinks(gts, face & (blink > cfg["gaze"]["blink_threshold"]))
    b0, b1 = np.searchsorted(b_start, ends - W), np.searchsorted(b_start, ends)
    nb = (b1 - b0).astype(float)
    face_min = np.maximum(n_face, 1) / np.maximum(n_gaze, 1) * W / 60
    F["blink_rate"] = np.where(has & (n_face >= 5), nb / np.maximum(face_min, 1e-6), np.nan)
    F["blink_dur"] = np.where(has & (nb > 0), _ratio(_wsum(_csum(b_dur), b0, b1), nb), np.nan)

    # 注视
    fs, fe, _, _ = detect_fixations(gts[pos], g["x"][pos], g["y"][pos],
                                    a["fixation_dispersion_px"], a["fixation_min_s"])
    if len(fs):
        f0 = np.searchsorted(fe, ends - W)  # 第一个结束于窗口起点之后的注视
        f1 = np.searchsorted(fs, ends)      # 开始于窗口终点之前
        for k in np.nonzero(pos_ok)[0]:
            if f1[k] <= f0[k]:
                F["fix_count"][k], F["fix_mean"][k], F["fix_max"][k] = 0, 0, 0
                continue
            dur = np.minimum(fe[f0[k]:f1[k]], ends[k]) - np.maximum(fs[f0[k]:f1[k]], ends[k] - W)
            F["fix_count"][k] = len(dur)
            F["fix_mean"][k] = dur.mean()
            F["fix_max"][k] = dur.max()
    else:
        F["fix_count"][pos_ok], F["fix_mean"][pos_ok], F["fix_max"][pos_ok] = 0, 0, 0
