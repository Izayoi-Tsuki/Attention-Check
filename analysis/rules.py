"""规则状态机：按优先级给每个窗口判一个状态，再去掉持续时间过短的状态（去抖）。

窗口特征见 features.py。没有视线数据时，只能依据前台窗口和键鼠活跃度判断，
此时"专注"仅表示"在任务相关窗口内且有操作"。
"""
import numpy as np

# 状态 → (中文名, 颜色)。顺序即图例顺序
STATES = {
    "FOCUSED": ("专注", "#2e7d32"),
    "NEUTRAL": ("一般", "#a5d6a7"),
    "OFF_TASK_APP": ("任务外应用", "#ef6c00"),
    "LOOK_AWAY": ("视线离开屏幕", "#8e24aa"),
    "ZONED_OUT": ("发呆", "#c62828"),
    "DROWSY": ("犯困", "#6d4c41"),
    "INACTIVE": ("无操作", "#bdbdbd"),
    "AWAY": ("离开", "#546e7a"),
    "NO_DATA": ("未记录", "#ffffff"),
}
DISTRACTED = ("OFF_TASK_APP", "LOOK_AWAY", "ZONED_OUT", "DROWSY")


def state_name(s: str) -> str:
    return STATES.get(s, (s,))[0]


def classify_window(f: dict, r: dict, min_coverage: float = 0.5) -> str:
    """f：单个窗口的特征（标量字典）；r：config 中 rules 段。"""
    if f["coverage"] < min_coverage:
        return "NO_DATA"
    has_face = np.isfinite(f["face_ratio"])
    has_pos = np.isfinite(f["offscreen_ratio"])

    if has_face:
        if f["face_ratio"] < r["away_no_face_ratio"]:
            return "AWAY"
        if f["perclos"] >= r["drowsy_perclos"] or f["blink_dur"] >= r["drowsy_blink_s"]:
            return "DROWSY"
    elif f["idle_end"] >= r["away_idle_s"]:
        return "AWAY"

    if f["off_task_ratio"] >= r["off_task_ratio"]:
        return "OFF_TASK_APP"

    if has_pos:
        if f["offscreen_ratio"] >= r["look_away_ratio"]:
            return "LOOK_AWAY"
        if f["input_gap"] >= r["zoned_input_gap_s"] and (
                f["fix_max"] >= r["zoned_max_fixation_s"] or f["dispersion"] <= r["zoned_dispersion_px"]):
            return "ZONED_OUT"
        return "FOCUSED" if f["in_window_ratio"] >= r["focus_in_window_ratio"] else "NEUTRAL"

    if f["idle_end"] >= r["inactive_gap_s"]:
        return "INACTIVE"
    return "FOCUSED"


def classify_all(F: dict, cfg: dict) -> np.ndarray:
    n = len(F["end"])
    keys = [k for k, v in F.items() if isinstance(v, np.ndarray) and len(v) == n]
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = classify_window({k: F[k][i] for k in keys}, cfg["rules"])
    return out


def runs(states: np.ndarray):
    """连续相同状态的区间 [(state, i0, i1)]，i1 不含。"""
    out = []
    i, n = 0, len(states)
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        out.append((states[i], i, j))
        i = j
    return out


def debounce(states: np.ndarray, min_windows: int) -> np.ndarray:
    """持续不足 min_windows 个窗口的状态并入前一个状态（离线分析，可以回溯修改）。NO_DATA 不参与。"""
    s = states.copy()
    if min_windows <= 1 or len(s) == 0:
        return s
    for state, i0, i1 in runs(s):
        if state == "NO_DATA" or i1 - i0 >= min_windows:
            continue
        prev = s[i0 - 1] if i0 > 0 else None
        nxt = s[i1] if i1 < len(s) else None
        fill = prev if prev not in (None, "NO_DATA") else nxt
        if fill not in (None, "NO_DATA"):
            s[i0:i1] = fill
    return s


def label_states(F: dict, cfg: dict):
    raw = classify_all(F, cfg)
    return raw, debounce(raw, int(cfg["rules"]["debounce_windows"]))
