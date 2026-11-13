"""每日 HTML 报告（图表以 base64 内嵌，单文件，可直接用浏览器打开）。

命令行：  .venv\\Scripts\\python.exe -m analysis.report [--date 2026-01-31] [--db 路径] [--open]
"""
import argparse
import base64
import datetime as dt
import html
import io
import webbrowser
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from storage.db import ANSWERS  # noqa: E402

from .distraction import REGIONS, episodes, find_events, focus_streaks, rank_sources  # noqa: E402
from .features import compute_windows, load_range  # noqa: E402
from .rules import STATES, label_states, state_name  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CATEGORY_NAMES = {"task": "工作", "comms": "通讯", "distraction": "干扰", "neutral": "中性", "": ""}
STREAK_BUCKETS = [(0, 5, "< 5 分钟"), (5, 15, "5–15 分钟"), (15, 30, "15–30 分钟"),
                  (30, 60, "30–60 分钟"), (60, 1e9, "≥ 60 分钟")]


# ---------- 计算 ----------
def analyze(cfg: dict, date: dt.date, db_path: str = None) -> dict:
    t0 = dt.datetime.combine(date, dt.time()).timestamp()
    t1 = t0 + 86400
    d = load_range(db_path or cfg["storage"]["db_path"], t0, t1)
    F = compute_windows(d, cfg)
    step = float(cfg["analysis"]["step_s"])
    if len(F["end"]) == 0:
        return dict(d=d, F=F, empty=True)
    raw, states = label_states(F, cfg)
    eps = episodes(F, states, step)
    events = find_events(d, F, states, cfg)
    return dict(d=d, F=F, raw=raw, states=states, eps=eps, events=events,
                sources=rank_sources(events), streaks=focus_streaks(eps), step=step, empty=False)


def state_at(R: dict, ts: float) -> str:
    """ts 之前最后一个完整窗口的状态（自评弹窗本身会改变前台窗口，所以取弹出前的状态）。"""
    ends = R["F"]["end"]
    i = int(np.searchsorted(ends, ts, side="right")) - 1
    if i < 0 or ts - ends[i] > 2 * R["step"]:
        return "NO_DATA"
    return R["states"][i]


# ---------- 图表 ----------
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _hour_ticks(ax, h0, h1):
    ticks = np.arange(np.floor(h0), np.ceil(h1) + 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t):02d}:00" for t in ticks])
    ax.set_xlim(h0, h1)


def chart_timeline(R: dict) -> str:
    d, F, step = R["d"], R["F"], R["step"]
    eps = [e for e in R["eps"] if e["state"] != "NO_DATA"]
    h = lambda t: (t - d.t0) / 3600  # noqa: E731
    h0 = h(eps[0]["start"]) if eps else 0
    h1 = h(eps[-1]["end"]) if eps else 24
    h0, h1 = np.floor(h0 * 4) / 4, np.ceil(h1 * 4) / 4
    if h1 - h0 < 1:
        h1 = h0 + 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 4.2), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1, 1.6], hspace=0.12))
    present = []
    for s, (name, color) in STATES.items():
        bars = [(h(e["start"]), (e["end"] - e["start"]) / 3600) for e in eps if e["state"] == s]
        if bars:
            ax1.broken_barh(bars, (0, 1), facecolors=color, label=name)
            present.append(s)
    ax1.set_yticks([])
    ax1.set_title("状态时间线", loc="left", fontsize=11)
    if present:
        ax1.legend(ncol=len(present), loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False, fontsize=9)

    # 每 15 分钟的专注率
    ends, states = F["end"], R["states"]
    b = ((ends - step - d.t0) // 900).astype(int)
    rec = states != "NO_DATA"
    present_mask = rec & (states != "AWAY")
    xs, ys, cover = [], [], []
    for k in np.unique(b[rec]):
        m = b == k
        n_present = (m & present_mask).sum()
        xs.append(k / 4)
        ys.append(100 * (m & (states == "FOCUSED")).sum() / n_present if n_present else 0)
        cover.append((m & rec).sum() * step / 900)
    colors = [(0.18, 0.49, 0.2, 0.3 + 0.7 * min(c, 1)) for c in cover]
    ax2.bar(np.array(xs) + 0.125, ys, width=0.24, color=colors)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("专注率 %")
    ax2.set_title("每 15 分钟专注率（不含离开时间；颜色越浅表示该时段记录越少）", loc="left", fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    _hour_ticks(ax2, h0, h1)
    return _png(fig)


def chart_streaks(R: dict) -> str:
    mins = np.array(R["streaks"]) / 60
    labels = [b[2] for b in STREAK_BUCKETS]
    counts = [int(((mins >= lo) & (mins < hi)).sum()) for lo, hi, _ in STREAK_BUCKETS]
    totals = [float(mins[(mins >= lo) & (mins < hi)].sum()) for lo, hi, _ in STREAK_BUCKETS]
    fig, ax = plt.subplots(figsize=(6.5, 3))
    bars = ax.bar(labels, totals, color="#2e7d32")
    for rect, c, t in zip(bars, counts, totals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(), f"{c} 段\n{t:.0f} 分钟",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("累计专注分钟")
    ax.set_ylim(0, max(totals + [1]) * 1.35)
    ax.set_title("连续专注时长分布", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    return _png(fig)


def chart_heatmap(R: dict):
    d = R["d"]
    g = d.gaze
    sw, sh = d.screen
    ok = (g["valid"] == 1) & np.isfinite(g["x"]) & np.isfinite(g["y"])
    x, y = g["x"][ok], g["y"][ok]
    if len(x) < 50:
        return None, None
    inside = (x >= 0) & (x < sw) & (y >= 0) & (y < sh)
    H, _, _ = np.histogram2d(y[inside], x[inside], bins=[18, 32], range=[[0, sh], [0, sw]])
    fig, ax = plt.subplots(figsize=(7, 7 * sh / sw + 0.6))
    ax.imshow(H, extent=(0, sw, sh, 0), cmap="inferno", interpolation="bicubic", aspect="equal")
    grid = np.zeros((3, 3))
    for r in range(3):
        for c in range(3):
            m = inside & (x >= c * sw / 3) & (x < (c + 1) * sw / 3) & (y >= r * sh / 3) & (y < (r + 1) * sh / 3)
            grid[r, c] = m.sum() / len(x)
            ax.text((c + 0.5) * sw / 3, (r + 0.5) * sh / 3, f"{REGIONS[r][c]}\n{grid[r, c] * 100:.0f}%",
                    ha="center", va="center", color="white", fontsize=10,
                    bbox=dict(facecolor="black", alpha=0.35, edgecolor="none"))
    for k in (1, 2):
        ax.axvline(k * sw / 3, color="white", alpha=0.4, lw=0.8)
        ax.axhline(k * sh / 3, color="white", alpha=0.4, lw=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    off = 1 - inside.mean()
    ax.set_title(f"视线分布热力图（屏幕外 {off * 100:.0f}%）", loc="left", fontsize=11)
    return _png(fig), off


# ---------- HTML ----------
def _e(s) -> str:
    return html.escape(str(s))


def _hm(ts) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def _hms(ts) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _dur(sec) -> str:
    sec = int(round(sec))
    return f"{sec // 3600}小时{sec % 3600 // 60:02d}分" if sec >= 3600 else f"{sec // 60}分{sec % 60:02d}秒"


def _table(headers, rows, cls="") -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>'


def _img(b64) -> str:
    return f'<img src="data:image/png;base64,{b64}">'


def _card(label, value, sub="") -> str:
    return f'<div class="card"><div class="v">{_e(value)}</div><div class="l">{_e(label)}</div>' \
           f'<div class="s">{_e(sub)}</div></div>'


CSS = """
body{font-family:"Microsoft YaHei UI","Microsoft YaHei",sans-serif;margin:24px auto;max-width:1180px;color:#222;
     background:#fafafa;padding:0 16px}
h1{font-size:22px;margin-bottom:4px} h2{font-size:17px;margin:28px 0 8px;border-left:4px solid #2e7d32;padding-left:8px}
.muted{color:#777;font-size:13px} .note{background:#fff8e1;border:1px solid #ffe082;padding:8px 12px;border-radius:6px;
     font-size:13px;margin:10px 0}
.cards{display:flex;flex-wrap:wrap;gap:10px} .card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;
     padding:10px 14px;min-width:140px} .card .v{font-size:22px;font-weight:600} .card .l{font-size:13px;color:#555}
.card .s{font-size:12px;color:#999}
table{border-collapse:collapse;background:#fff;font-size:13px;margin:6px 0} th,td{border:1px solid #e0e0e0;
     padding:4px 10px;text-align:left} th{background:#f1f8e9} td.num,th.num{text-align:right}
img{max-width:100%;background:#fff;border:1px solid #eee;border-radius:6px}
.row{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
"""


def build_html(R: dict, cfg: dict, date: dt.date) -> str:
    d = R["d"]
    parts = [f"<h1>注意力日报 · {date.isoformat()}</h1>",
             f'<div class="muted">生成于 {dt.datetime.now().strftime("%Y-%m-%d %H:%M")} · 数据仅保存在本机</div>']
    if R["empty"]:
        parts.append('<div class="note">这一天没有任何记录。</div>')
        return _wrap(parts)

    F, states, step = R["F"], R["states"], R["step"]
    dur = {s: float((states == s).sum() * step) for s in STATES}
    recorded = sum(v for s, v in dur.items() if s != "NO_DATA")
    present = recorded - dur["AWAY"]
    focus = dur["FOCUSED"]
    events = R["events"]
    rets = [e["return_s"] for e in events if e["return_s"] is not None]
    streaks = R["streaks"]
    probes = [p for p in d.probes if p[3] is not None]
    answered = [p for p in probes if p[2] == "random" and p[3] > 0]
    missed = [p for p in probes if p[2] == "random" and p[3] <= 0]
    selfs = [p for p in probes if p[2] == "self"]

    # 概要
    parts.append("<h2>概要</h2><div class='cards'>")
    parts.append(_card("记录时长", _dur(recorded), f"其中离开 {_dur(dur['AWAY'])}"))
    parts.append(_card("专注时长", _dur(focus), f"占在座时间 {focus / present * 100:.0f}%" if present else ""))
    parts.append(_card("分心次数", len(events), f"每小时 {len(events) / (present / 3600):.1f} 次"
                       if present >= 600 else ""))
    parts.append(_card("平均回到专注", f"{np.median(rets) / 60:.1f} 分钟" if rets else "—",
                       f"中位数，{len(rets)} 次可计" if rets else ""))
    parts.append(_card("最长连续专注", _dur(max(streaks)) if streaks else "—"))
    parts.append(_card("自评", f"{len(answered)} 次", f"未回应 {len(missed)} · 自察 {len(selfs)}"))
    parts.append("</div>")
    if not d.has_gaze:
        cam_note = any(ev[2] == "camera_unavailable" for ev in d.events)
        parts.append('<div class="note">今天没有视线数据' + ("（摄像头不可用）" if cam_note else "") +
                     "：状态仅依据前台窗口与键鼠活跃度判断。“专注”表示在任务相关窗口内且有操作，"
                     "无法识别发呆、视线离开屏幕、犯困，阅读时长时间不操作会被判为“无操作”。</div>")

    # 状态构成
    rows = [[f'<span class="sw" style="background:{STATES[s][1]}"></span>{_e(STATES[s][0])}', _dur(v),
             f"{v / recorded * 100:.1f}%"] for s, v in dur.items() if v > 0 and s != "NO_DATA"]
    parts.append("<h2>状态时间线</h2>")
    parts.append(_img(chart_timeline(R)))
    parts.append(_table(["状态", "时长", "占比"], rows))

    # 会话
    if d.sessions:
        srows = []
        for sid, start, end, task in d.sessions:
            end = end or float(F["end"][-1])
            srows.append([_e(task), f"{_hm(start)} – {_hm(end)}", _dur(end - start)])
        parts.append("<h2>任务会话</h2>" + _table(["任务", "时间", "时长"], srows))

    # 干扰源排行
    parts.append("<h2>干扰源排行</h2>")
    if R["sources"]:
        srows = [[_e(s["kind"]), _e(s["trigger"]), _e(CATEGORY_NAMES.get(s["category"], s["category"])),
                  s["count"], "—" if s["mean_return_min"] is None else f"{s['mean_return_min']:.1f}",
                  f"{s['total_return_min']:.0f}"] for s in R["sources"][:20]]
        parts.append(_table(["类型", "触发源", "类别", "次数", "平均回到专注（分钟）", "累计（分钟）"], srows))
        parts.append(f'<div class="muted">按 次数 × 平均回到专注耗时 排序。分心后离开座位、停止记录或超过 '
                     f'{cfg["rules"]["return_cap_min"]} 分钟未回到专注的，不计入回归耗时。</div>')
    else:
        parts.append('<div class="muted">没有检测到从专注转入分心的事件。</div>')

    # 专注时段分布 + 视线热力图
    parts.append("<h2>专注时段分布</h2><div class='row'>")
    parts.append(_img(chart_streaks(R)) if streaks else '<div class="muted">今天没有专注片段。</div>')
    if d.has_gaze:
        img, _ = chart_heatmap(R)
        if img:
            parts.append(_img(img))
    parts.append("</div>")

    # 应用用时
    c = d.ctx
    sample_s = float(cfg["context"]["sample_seconds"])
    usage = defaultdict(lambda: [0.0, 0.0, ""])
    for lab, cat, on in zip(c["label"], c["category"], c["on_task"]):
        u = usage[lab or "（未知）"]
        u[0] += sample_s
        u[1] += sample_s * on
        u[2] = cat
    urows = [[_e(k), _e(CATEGORY_NAMES.get(v[2], v[2])), _dur(v[0]), f"{v[1] / v[0] * 100:.0f}%"]
             for k, v in sorted(usage.items(), key=lambda kv: -kv[1][0])[:20]]
    parts.append("<h2>前台应用用时（前 20）</h2>")
    parts.append(_table(["应用 / 网站", "类别", "前台时长", "任务内占比"], urows))

    # 自评对照
    parts.append("<h2>自评与规则判定对照</h2>")
    if answered or selfs:
        groups = [(ANSWERS[a], [p for p in answered if p[3] == a]) for a in sorted(ANSWERS)]
        groups.append(("自察热键（发现走神）", selfs))
        cols = [s for s in STATES if any(state_at(R, p[0]) == s for _, ps in groups for p in ps)]
        prow = []
        for name, ps in groups:
            if not ps:
                continue
            cnt = defaultdict(int)
            for p in ps:
                cnt[state_at(R, p[0])] += 1
            prow.append([_e(name), len(ps)] + [cnt.get(s, "") for s in cols])
        parts.append(_table(["自评答案", "次数"] + [state_name(s) for s in cols], prow))
        focus_ans = [p for p in answered if p[3] == 1]
        mind_ans = [p for p in answered if p[3] in (2, 3, 5)] + selfs
        notes = []
        if focus_ans:
            hit = sum(state_at(R, p[0]) in ("FOCUSED", "NEUTRAL") for p in focus_ans)
            notes.append(f"回答“专注”时，规则判为专注/一般的比例 {hit / len(focus_ans) * 100:.0f}%")
        if mind_ans:
            hit = sum(state_at(R, p[0]) not in ("FOCUSED", "NO_DATA") for p in mind_ans)
            notes.append(f"回答走神/被吸引/犯困（含自察）时，规则判为非专注的比例 {hit / len(mind_ans) * 100:.0f}%")
        rts = [p[4] for p in answered if p[4]]
        if rts:
            notes.append(f"自评平均响应 {np.mean(rts) / 1000:.1f} 秒")
        parts.append('<div class="muted">' + "；".join(notes) +
                     "。第二个比例偏低说明规则漏掉了“看着屏幕在走神”，这正是后续个人模型要学习的部分。</div>")
    else:
        parts.append('<div class="muted">今天还没有自评记录。</div>')

    # 分心事件明细
    if events:
        erows = []
        for e in events[-100:]:
            erows.append([_hms(e["ts"]), _e(e["kind_name"]), _e(e["trigger"])]
                         + ([_e(e["gaze_region"] or "—")] if d.has_gaze else [])
                         + [_dur(e["duration_s"]), "—" if e["return_s"] is None else f"{e['return_s'] / 60:.1f}"])
        heads = ["时间", "类型", "触发源"] + (["事发前视线"] if d.has_gaze else []) + ["持续", "回到专注（分钟）"]
        parts.append(f"<h2>分心事件明细（{'最近 100 条' if len(events) > 100 else f'共 {len(events)} 条'}）</h2>")
        parts.append(_table(heads, erows))
    return _wrap(parts)


def _wrap(parts) -> str:
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>注意力日报</title>'
            f"<style>{CSS}</style></head><body>" + "\n".join(parts) + "</body></html>")


def generate_report(cfg: dict, date: dt.date, db_path: str = None, out_dir: str = None) -> Path:
    R = analyze(cfg, date, db_path)
    out = Path(out_dir or cfg["storage"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = (out / f"{date.isoformat()}.html").resolve()
    path.write_text(build_html(R, cfg, date), encoding="utf-8")
    return path


def main():
    from settings import load_config
    ap = argparse.ArgumentParser(description="生成注意力日报")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="YYYY-MM-DD，默认今天")
    ap.add_argument("--db", default=None, help="数据库路径，默认取 config.yaml")
    ap.add_argument("--out", default=None, help="输出目录，默认取 config.yaml")
    ap.add_argument("--open", action="store_true", help="生成后用浏览器打开")
    args = ap.parse_args()
    path = generate_report(load_config(), dt.date.fromisoformat(args.date), args.db, args.out)
    print(path)
    if args.open:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    main()
