"""第 3 步自检：按脚本生成两天模拟数据（一天有视线、一天无摄像头），跑完整分析并生成报告，
再把规则判定与脚本中的真实行为对照。

运行：  .venv\\Scripts\\python.exe selftest_report.py [--open]
数据写入 data/selftest/selftest.db，与正式数据库分开。
"""
import argparse
import datetime as dt
import sys
import webbrowser
from pathlib import Path

import numpy as np

from analysis.report import analyze, generate_report
from analysis.rules import state_name
from capture.context import Classifier
from settings import ROOT, load_config
from storage.db import COLUMNS, connect

SW, SH = 2560, 1440
FPS = 10
NOISE = 25  # 视线噪声标准差（px，平滑后），可用 --noise 修改
OUT = ROOT / "data" / "selftest"
DAY_GAZE = dt.date(2026, 1, 5)
DAY_NOCAM = dt.date(2026, 1, 6)

# 行为 → (exe, 标题, 窗口矩形)
APPS = {
    "focus": ("Code.exe", "main.py - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
    "read": ("Code.exe", "README.md - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
    "bili": ("msedge.exe", "【4K】城市夜景 - 哔哩哔哩 - Microsoft Edge", (0, 0, SW, SH - 40)),
    "wechat": ("WeChat.exe", "微信", (700, 200, 1900, 1240)),
    "zoned": ("Code.exe", "main.py - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
    "look_away": ("Code.exe", "main.py - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
    "drowsy": ("Code.exe", "main.py - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
    "away": ("Code.exe", "main.py - attention-tracker - Visual Studio Code", (0, 0, SW, SH - 40)),
}
# 每秒输入次数的泊松均值：keys, clicks, moves, scrolls
INPUT_RATE = {
    "focus": (3.0, 0.1, 4, 0.1), "read": (0.0, 0.05, 1, 0.25), "bili": (0.05, 0.1, 6, 0.4),
    "wechat": (2.0, 0.2, 3, 0.1), "drowsy": (0.4, 0.02, 0.5, 0.0),
    "zoned": (0, 0, 0, 0), "look_away": (0, 0, 0, 0), "away": (0, 0, 0, 0),
}
TRUTH = {"focus": "FOCUSED", "read": "FOCUSED", "bili": "OFF_TASK_APP", "wechat": "OFF_TASK_APP",
         "zoned": "ZONED_OUT", "look_away": "LOOK_AWAY", "drowsy": "DROWSY", "away": "AWAY"}
# 无摄像头时规则能区分的只有这些；离开的前 away_idle_s 秒只能判为"无操作"
TRUTH_NOCAM = {"focus": "FOCUSED", "read": "FOCUSED", "bili": "OFF_TASK_APP", "wechat": "OFF_TASK_APP",
               "away": ("AWAY", "INACTIVE")}
DISTRACTIONS = [("bili", 120, 360), ("wechat", 60, 180), ("zoned", 60, 120),
                ("look_away", 40, 90), ("drowsy", 60, 120)]


def script_day(rng, sessions):
    """返回 [(kind, start, end)]，时间为 Unix 秒。"""
    segs = []
    for n_session, (s0, s1) in enumerate(sessions):
        t = s0
        first = True
        while t < s1:
            for kind, dur in [("focus", rng.uniform(420, 1200)), ("read", rng.uniform(60, 240)),
                              ("focus", rng.uniform(180, 600))]:
                segs.append((kind, t, min(t + dur, s1)))
                t += dur
                if t >= s1:
                    break
            if t >= s1:
                break
            if (n_session == 1 and first) or rng.random() < 0.12:
                kind, lo, hi = "away", 300, 600
            else:
                kind, lo, hi = DISTRACTIONS[rng.choice(len(DISTRACTIONS), p=[0.35, 0.25, 0.15, 0.15, 0.1])]
            first = False
            dur = rng.uniform(lo, hi)
            segs.append((kind, t, min(t + dur, s1)))
            t += dur
    return segs


def gen_gaze(rng, kind, t0, t1, rect):
    n = int((t1 - t0) * FPS)
    ts = t0 + np.arange(n) / FPS
    face = np.ones(n)
    bl = rng.uniform(0.05, 0.2, n)
    if kind == "away":
        face[:] = 0
        x = y = np.full(n, np.nan)
        bl[:] = 0
    elif kind == "zoned":
        x = rng.normal(1300, NOISE, n)
        y = rng.normal(650, NOISE, n)
    elif kind == "look_away":
        x = rng.normal(1200, 150, n)
        y = rng.normal(SH + 350, 120, n)
    else:
        # 一连串注视点，每个 0.2–0.6s，落在窗口内
        x, y = np.empty(n), np.empty(n)
        i = 0
        l, t, r, b = rect
        while i < n:
            k = int(rng.uniform(2, 6))
            x[i:i + k] = rng.uniform(l + 40, r - 40)
            y[i:i + k] = rng.uniform(t + 40, b - 40)
            i += k
        x += rng.normal(0, NOISE, n)
        y += rng.normal(0, NOISE, n)
    if kind != "away":
        # 眨眼：约 15 次/分钟，每次 2 帧；犯困时闭眼更久更频繁
        rate, length = (25, 6) if kind == "drowsy" else (15, 2)
        starts = np.nonzero(rng.random(n) < rate / 60 / FPS)[0]
        for s in starts:
            bl[s:s + length] = rng.uniform(0.75, 0.95)
    br = np.clip(bl + rng.normal(0, 0.03, n), 0, 1)
    valid = ((face == 1) & (np.fmax(bl, br) <= 0.45)).astype(float)
    x = np.where(valid == 1, x, np.nan)
    y = np.where(valid == 1, y, np.nan)
    yaw = np.where(face == 1, rng.normal(0, 2, n), 0)
    pitch = np.where(face == 1, rng.normal(-5 if kind != "look_away" else -25, 2, n), 0)
    return ts, face, valid, x, y, bl, br, yaw, pitch


def build_db(db_path: Path, cfg: dict, date: dt.date, with_gaze: bool, seed: int):
    rng = np.random.default_rng(seed)
    conn = connect(str(db_path))
    day0 = dt.datetime.combine(date, dt.time()).timestamp()
    for table in ("sessions", "calibrations", "gaze", "context", "probes", "events"):
        conn.execute(f"DELETE FROM {table} WHERE {'start' if table == 'sessions' else 'ts'} >= ? "
                     f"AND {'start' if table == 'sessions' else 'ts'} < ?", (day0, day0 + 86400))
    sessions = [(day0 + 9 * 3600, day0 + 12 * 3600), (day0 + 14 * 3600, day0 + 17.5 * 3600)]
    segs = script_day(rng, sessions)
    clf = Classifier(cfg["categories"])
    clf.set_task_apps("")

    calib_id = None
    if with_gaze:
        calib_id = conn.execute("INSERT INTO calibrations(ts, val_error_px, cv_error_px, alpha, screen_w, screen_h, "
                                "model) VALUES(?,?,?,?,?,?,?)", (sessions[0][0], 95, 80, 10, SW, SH, b"")).lastrowid
    sids = []
    for s0, s1 in sessions:
        sid = conn.execute("INSERT INTO sessions(start, end, task, task_apps, calib_id) VALUES(?,?,?,?,?)",
                           (s0, s1, "写注意力追踪的分析模块", "", calib_id)).lastrowid
        sids.append(sid)
        conn.execute("INSERT INTO events VALUES(?,?,?,?)", (s0, sid, "session_start", ""))
        if not with_gaze:
            conn.execute("INSERT INTO events VALUES(?,?,?,?)", (s0 + 20, sid, "camera_unavailable", "未找到摄像头"))

    def sid_at(t):
        return sids[0] if t < sessions[0][1] else sids[1]

    ctx_rows, gaze_rows = [], []
    last_input = sessions[0][0]
    for kind, t0, t1 in segs:
        exe, title, rect = APPS[kind]
        cat, label = clf.classify(exe, title)
        on_task = int(clf.on_task(exe, title, cat))
        rates = INPUT_RATE[kind]
        for t in np.arange(np.ceil(t0), t1, 1.0):
            k, c, m, s = (int(v) for v in rng.poisson(rates))
            if k + c + m + s:
                last_input = t + rng.uniform(0, 1)
            idle = max(0.0, t + 1 - last_input)
            ctx_rows.append((t + 1, sid_at(t), exe, title, cat, label, on_task, *rect, k, c, m, s, idle))
        if with_gaze:
            ts, face, valid, x, y, bl, br, yaw, pitch = gen_gaze(rng, kind, t0, t1, rect)
            for i in range(len(ts)):
                gaze_rows.append((ts[i], sid_at(ts[i]), int(face[i]), int(valid[i]),
                                  None if np.isnan(x[i]) else float(x[i]), None if np.isnan(y[i]) else float(y[i]),
                                  float(bl[i]), float(br[i]), float(yaw[i]), float(pitch[i]), 0.0,
                                  0.0, 0.0, 60.0, calib_id, None))

    # 自评：每 8–20 分钟一次，按当时的真实行为作答；专注时有 15% 是"看着屏幕在走神"
    probe_rows = []
    for s0, s1 in sessions:
        t = s0 + rng.uniform(480, 1200)
        while t < s1:
            kind = next(k for k, a, b in segs if a <= t < b)
            ans = {"focus": 1, "read": 1, "bili": 3, "wechat": 3, "zoned": 2, "look_away": 2, "drowsy": 5,
                   "away": -1}[kind]
            if ans == 1 and rng.random() < 0.15:
                ans = 2
            probe_rows.append((t, sid_at(t), "random", ans, None if ans < 0 else int(rng.uniform(1500, 5000)),
                               0))
            t += rng.uniform(480, 1200)
    for kind, t0, t1 in segs:
        if kind == "zoned" and rng.random() < 0.5:
            probe_rows.append((t1 - 5, sid_at(t0), "self", 2, None, None))

    sql = {t: f"INSERT INTO {t}({','.join(c)}) VALUES({','.join('?' * len(c))})" for t, c in COLUMNS.items()}
    conn.executemany(sql["context"], ctx_rows)
    conn.executemany(sql["gaze"], gaze_rows)
    conn.executemany(sql["probes"], probe_rows)
    conn.commit()
    conn.close()
    return segs, len(ctx_rows), len(gaze_rows)


def evaluate(R, segs, truth_map, window_s):
    """只评估整个窗口都处于同一种真实行为的窗口。"""
    ends, states = R["F"]["end"], R["states"]
    starts = np.array([a for _, a, _ in segs])
    stops = np.array([b for _, _, b in segs])
    hit = total = 0
    confusion = {}
    for e, s in zip(ends, states):
        i = np.searchsorted(starts, e - window_s, side="right") - 1
        if i < 0 or stops[i] < e or segs[i][0] not in truth_map:
            continue
        t = truth_map[segs[i][0]]
        ok = s in t if isinstance(t, tuple) else s == t
        t = t[0] if isinstance(t, tuple) else t
        total += 1
        hit += ok
        confusion.setdefault(t, {}).setdefault(s, 0)
        confusion[t][s] += 1
    return hit, total, confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--noise", type=float, default=NOISE)
    args = ap.parse_args()
    globals()["NOISE"] = args.noise
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    db = OUT / "selftest.db"
    ok = True
    for date, with_gaze, truth in ((DAY_GAZE, True, TRUTH), (DAY_NOCAM, False, TRUTH_NOCAM)):
        segs, nc, ng = build_db(db, cfg, date, with_gaze, seed=date.toordinal())
        n_dist = sum(k in ("bili", "wechat", "zoned", "look_away", "drowsy") for k, _, _ in segs)
        R = analyze(cfg, date, str(db))
        hit, total, conf = evaluate(R, segs, truth, cfg["analysis"]["window_s"])
        acc = hit / max(total, 1)
        print(f"\n=== {date}（{'有视线' if with_gaze else '无摄像头'}）  context {nc} 行  gaze {ng} 行")
        print(f"窗口 {len(R['F']['end'])} 个；规则与真实行为一致 {hit}/{total} = {acc:.1%}")
        for t, row in conf.items():
            items = "  ".join(f"{state_name(s)} {n}" for s, n in sorted(row.items(), key=lambda kv: -kv[1]))
            print(f"  真实 {state_name(t):<6} → {items}")
        print(f"脚本中的分心片段 {n_dist} 个；检测到分心事件 {len(R['events'])} 个")
        for s in R["sources"][:5]:
            mr = "—" if s["mean_return_min"] is None else f"{s['mean_return_min']:.1f} 分钟"
            print(f"  {s['kind']:<8} {s['trigger']:<20} {s['count']:>3} 次  平均回到专注 {mr}")
        path = generate_report(cfg, date, str(db), str(OUT))
        print(f"报告：{path}  ({path.stat().st_size // 1024} KB)")
        need = 0.85 if with_gaze else 0.9
        if acc < need:
            print(f"!! 一致率低于 {need:.0%}")
            ok = False
        if args.open:
            webbrowser.open(path.as_uri())
    print("\n全部通过" if ok else "\n存在未通过项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
