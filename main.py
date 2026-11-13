"""注意力追踪主程序：持续记录屏幕上下文、自评标签，以及（有摄像头时）视线数据。

运行：  .venv\\Scripts\\python.exe main.py
没有摄像头时自动降级为只记录上下文与自评。所有数据只写入本地 data/attention.db。
"""
import datetime as dt
import pickle
import queue
import threading
import time
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from typing import Optional

import numpy as np
from pynput import keyboard

from capture import screen
from capture.calibration import CalibrationWindow, posture_warning
from capture.context import Classifier, ContextSampler, InputMonitor
from capture.gaze import GazeTracker
from capture.overlay import GazeOverlay
from capture.probe import ProbePopup, ProbeScheduler, Toast
from settings import load_config
from storage.db import Database

CATEGORY_NAMES = {"task": "工作", "comms": "通讯", "distraction": "干扰", "neutral": "中性"}


class SessionDialog:
    """声明当前任务。task_apps 用于判断"是否在做正事"，留空则按 config 中的 task 类别判断。"""

    def __init__(self, root, last_task: str, last_apps: str, on_done):
        self.on_done = on_done
        w = self.win = tk.Toplevel(root)
        w.title("开始一段记录")
        w.attributes("-topmost", True)
        w.resizable(False, False)
        f = ttk.Frame(w, padding=14)
        f.pack()
        ttk.Label(f, text="当前要做的事（简短描述）").pack(anchor="w")
        self.task = ttk.Entry(f, width=48)
        self.task.insert(0, last_task)
        self.task.pack(fill="x", pady=(2, 10))
        ttk.Label(f, text="与这件事相关的程序或窗口标题关键词，用逗号分隔\n"
                          "例如：Code.exe, Qoder.exe, 论文\n留空 = 按 config.yaml 中的 task 类别判断",
                  foreground="#555").pack(anchor="w")
        self.apps = ttk.Entry(f, width=48)
        self.apps.insert(0, last_apps)
        self.apps.pack(fill="x", pady=(2, 12))
        bar = ttk.Frame(f)
        bar.pack(fill="x")
        ttk.Button(bar, text="开始记录", command=self._ok).pack(side="right")
        ttk.Button(bar, text="取消", command=self._cancel).pack(side="right", padx=6)
        w.bind("<Return>", lambda e: self._ok())
        w.protocol("WM_DELETE_WINDOW", self._cancel)
        w.update_idletasks()
        sw, sh = screen.screen_size()
        w.geometry(f"+{(sw - w.winfo_reqwidth()) // 2}+{(sh - w.winfo_reqheight()) // 3}")
        screen.force_foreground(w)
        self.task.focus_set()

    def _ok(self):
        task, apps = self.task.get().strip() or "未命名任务", self.apps.get().strip()
        self.win.destroy()
        self.on_done(task, apps)

    def _cancel(self):
        self.win.destroy()
        self.on_done(None, None)


class App:
    def __init__(self):
        screen.set_dpi_awareness()
        self.cfg = load_config()
        self.sw, self.sh = screen.screen_size()
        self.db = Database(self.cfg["storage"]["db_path"])
        self.clock_offset = time.time() - time.monotonic()  # monotonic → Unix 时间

        self.session_id: Optional[int] = None
        self.session_start = 0.0
        self.task = ""
        self.paused = False
        self.calib_id: Optional[int] = None
        self.cal_info: Optional[dict] = None
        self.mapper = None
        self.tracker: Optional[GazeTracker] = None
        self.camera_status = "未启用" if not self.cfg["camera"]["enabled"] else "启动中……"
        self.camera_ok = False  # 摄像头是否成功运行过（决定暂停恢复后是否重开）
        self.calibrating = False
        self.popup: Optional[ProbePopup] = None
        self.ui_queue: queue.Queue = queue.Queue()  # 其他线程 → 界面线程

        self.root = tk.Tk()
        self.root.title("注意力追踪")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.withdraw()

        self.classifier = Classifier(self.cfg["categories"])
        self.inputs = InputMonitor(on_click=self._on_click)
        self.sampler = ContextSampler(self.cfg, self.inputs, self.classifier, self._on_context)
        self.sampler.paused = True
        self.scheduler = ProbeScheduler(self.cfg)
        self.hotkeys = keyboard.GlobalHotKeys(
            {self.cfg["probes"]["self_hotkey"]: lambda: self.ui_queue.put("self_report")})
        self.overlay: Optional[GazeOverlay] = None

        today = dt.datetime.combine(dt.date.today(), dt.time()).timestamp()
        self.probe_counts = dict(self.db.query(
            "SELECT source, COUNT(*) FROM probes WHERE ts >= ? AND answer > 0 GROUP BY source", (today,)))

        self._build_panel()
        last = self.db.query("SELECT task, task_apps FROM sessions ORDER BY id DESC LIMIT 1")
        last_task, last_apps = last[0] if last else ("", "")
        SessionDialog(self.root, last_task, last_apps or "", self._on_first_session)

    # ---------- 会话 ----------
    def _on_first_session(self, task, apps):
        if task is None:
            self.db.close()
            self.root.destroy()
            return
        self.inputs.start()
        self.sampler.start()
        self.hotkeys.start()
        self._start_session(task, apps)
        self.root.deiconify()
        self.root.after(1000, self._tick)
        if self.cfg["camera"]["enabled"]:
            self._start_camera(calibrate=True)

    def _start_session(self, task, apps):
        self.classifier.set_task_apps(apps)
        self.task = task
        self.session_start = time.time()
        self.session_id = self.db.execute(
            "INSERT INTO sessions(start, task, task_apps, calib_id) VALUES(?,?,?,?)",
            (self.session_start, task, apps, self.calib_id))
        self.db.event(self.session_id, "session_start", task)
        self.sampler.paused = self.paused
        self.scheduler.reschedule()

    def _end_session(self):
        if self.session_id is None:
            return
        self.db.event(self.session_id, "session_end")
        self.db.execute("UPDATE sessions SET end=? WHERE id=?", (time.time(), self.session_id))
        self.session_id = None

    def switch_task(self):
        last = self.db.query("SELECT task, task_apps FROM sessions WHERE id=?", (self.session_id,))
        task, apps = last[0] if last else ("", "")

        def done(new_task, new_apps):
            if new_task is None:
                return
            self._end_session()
            self._start_session(new_task, new_apps)

        SessionDialog(self.root, task, apps or "", done)

    # ---------- 摄像头与校准 ----------
    def _start_camera(self, calibrate: bool):
        tr = GazeTracker(self.cfg)
        tr.mapper = self.mapper
        tr.on_sample = self._on_gaze
        tr.start()
        self.tracker = tr
        self.camera_status = "启动中……"
        t0 = time.monotonic()

        def wait():
            if tr is not self.tracker:
                return
            if tr.error:
                self.camera_status = f"不可用（{tr.error.split('：')[0]}），仅记录屏幕上下文"
                self.db.event(self.session_id, "camera_unavailable", tr.error)
                self.tracker = None
                return
            if tr.latest is None:
                if time.monotonic() - t0 > 20:
                    self.camera_status = "不可用（超时无画面），仅记录屏幕上下文"
                    tr.stop()
                    self.tracker = None
                    return
                self.root.after(300, wait)
                return
            self.camera_status = "运行中"
            self.camera_ok = True
            if self.overlay is None:
                self.overlay = GazeOverlay(self.root, self.sw, self.sh, self.cfg["overlay"]["dot_radius"])
                self.overlay.set_visible(False)
            if calibrate or self.mapper is None:
                self.calibrate()

        self.root.after(300, wait)

    def _stop_camera(self):
        if self.tracker:
            self.tracker.stop()
            self.tracker = None

    def calibrate(self):
        if self.calibrating or self.tracker is None or self.tracker.latest is None:
            return
        self.calibrating = True
        self._sync_overlay()
        self.root.withdraw()
        CalibrationWindow(self.root, self.tracker, self.cfg, (self.sw, self.sh), self._on_calibrated)

    def _on_calibrated(self, mapper, info):
        self.calibrating = False
        if mapper is not None:
            self.calib_id = self.db.execute(
                "INSERT INTO calibrations(ts, val_error_px, cv_error_px, alpha, screen_w, screen_h, model) "
                "VALUES(?,?,?,?,?,?,?)",
                (time.time(), info["val_error_px"], info["cv_error_px"], info["alpha"],
                 self.sw, self.sh, pickle.dumps(mapper)))
            self.db.execute("UPDATE sessions SET calib_id=? WHERE id=?", (self.calib_id, self.session_id))
            self.mapper = mapper
            self.cal_info = info
            if self.tracker:
                self.tracker.mapper = mapper
        self.root.deiconify()
        self._sync_overlay()

    def _sync_overlay(self):
        if self.overlay:
            on = (self.show_dot.get() and self.mapper is not None
                  and self.tracker is not None and not self.calibrating)
            was = self.overlay.visible
            self.overlay.set_visible(on)
            if on and not was:
                self._overlay_tick()

    def _overlay_tick(self):
        tr = self.tracker
        if not (self.overlay and self.overlay.visible and tr is not None):
            return
        g = tr.latest.gaze if tr.latest is not None else None
        xy = None if g is None else (min(max(g[0], 0), self.sw - 1), min(max(g[1], 0), self.sh - 1))
        self.overlay.update(xy, "")
        self.root.after(30, self._overlay_tick)

    # ---------- 采集回调（后台线程） ----------
    def _on_context(self, row: dict):
        if self.session_id is None:
            return
        self.db.insert("context", (
            row["ts"], self.session_id, row["exe"], row["title"], row["category"], row["key"], row["on_task"],
            *row["rect"], row["keys"], row["clicks"], row["mouse_moves"], row["scrolls"], row["idle_s"]))

    def _on_gaze(self, s):
        if self.session_id is None or self.paused or self.calibrating:
            return
        g = s.gaze
        t = s.t if s.t is not None else (None, None, None)
        self.db.insert("gaze", (
            s.ts + self.clock_offset, self.session_id, int(s.face), int(s.valid),
            None if g is None else float(g[0]), None if g is None else float(g[1]),
            s.blink_l, s.blink_r, s.yaw, s.pitch, s.roll,
            *(None if v is None else float(v) for v in t),
            self.calib_id if s.gaze is not None else None,
            None if s.feat is None else s.feat.astype(np.float32).tobytes()))

    def _on_click(self, x, y):
        tr, mapper = self.tracker, self.mapper
        if (tr is None or mapper is None or self.calibrating or self.paused
                or not self.cfg["implicit"]["enabled"]):
            return
        t = time.monotonic()
        w0, w1 = self.cfg["implicit"]["window"]
        frames = tr.samples_between(t - w0, t - w1)
        if len(frames) >= 2:
            mapper.add_implicit(np.median([f.feat for f in frames], axis=0), (x, y))

    # ---------- 自评 ----------
    def _show_probe(self, deferred_ms: int):
        shown_ts = time.time()

        def answered(code, response_ms):
            self.popup = None
            self.db.insert("probes", (shown_ts, self.session_id, "random", code, response_ms, deferred_ms))
            if code > 0:
                self.probe_counts["random"] = self.probe_counts.get("random", 0) + 1
            self.scheduler.reschedule()

        self.popup = ProbePopup(self.root, self.cfg["probes"]["timeout_s"], answered)

    def _self_report(self):
        if self.session_id is None or self.paused:
            return
        self.db.insert("probes", (time.time(), self.session_id, "self", 2, None, None))
        self.probe_counts["self"] = self.probe_counts.get("self", 0) + 1
        Toast(self.root, "已记录：发现自己走神")

    # ---------- 控制面板 ----------
    def _build_panel(self):
        f = ttk.Frame(self.root, padding=10)
        f.pack(fill="both", expand=True)
        self.status = tk.StringVar()
        ttk.Label(f, textvariable=self.status, font=("Microsoft YaHei UI", 10), justify="left",
                  wraplength=520).pack(anchor="w")
        self.warn = tk.StringVar()
        ttk.Label(f, textvariable=self.warn, foreground="#c00000", wraplength=520).pack(anchor="w", pady=(4, 0))
        self.info = tk.StringVar()
        ttk.Label(f, textvariable=self.info, foreground="#555", wraplength=520).pack(anchor="w")

        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=(10, 0))
        self.pause_btn = ttk.Button(bar, text="暂停", command=self.toggle_pause)
        self.pause_btn.pack(side="left")
        ttk.Button(bar, text="切换任务", command=self.switch_task).pack(side="left", padx=4)
        self.calib_btn = ttk.Button(bar, text="重新校准", command=self.calibrate)
        self.calib_btn.pack(side="left")
        self.show_dot = tk.BooleanVar(value=False)
        self.dot_chk = ttk.Checkbutton(bar, text="显示视线点", variable=self.show_dot, command=self._sync_overlay)
        self.dot_chk.pack(side="left", padx=6)
        ttk.Button(bar, text="退出", command=self.quit).pack(side="right")
        ttk.Button(bar, text="今日报告", command=self.open_report).pack(side="right", padx=4)
        hk = self.cfg["probes"]["self_hotkey"].replace("<", "").replace(">", "").title()
        ttk.Label(f, text=f"发现自己走神时按 {hk} 记录一次", foreground="#777").pack(anchor="w", pady=(8, 0))

        self.root.update_idletasks()
        self.root.geometry(f"+{self.sw - 600}+60")

    def toggle_pause(self):
        self.paused = not self.paused
        self.sampler.paused = self.paused
        self.db.event(self.session_id, "pause" if self.paused else "resume")
        self.pause_btn.configure(text="继续" if self.paused else "暂停")
        if self.paused:
            self._stop_camera()  # 暂停时关闭摄像头
            self._sync_overlay()
        else:
            self.scheduler.reschedule()
            if self.camera_ok:
                self._start_camera(calibrate=False)

    def open_report(self):
        self.info.set("正在生成今日报告……")

        def work():
            try:
                from analysis.report import generate_report
                path = generate_report(self.cfg, dt.date.today())
                webbrowser.open(path.as_uri())
                self.ui_queue.put(("info", f"报告已生成：{path}"))
            except Exception as e:
                self.ui_queue.put(("info", f"生成报告失败：{type(e).__name__}: {e}"))

        threading.Thread(target=work, daemon=True).start()

    def _tick(self):
        while not self.ui_queue.empty():
            msg = self.ui_queue.get_nowait()
            if msg == "self_report":
                self._self_report()
            elif isinstance(msg, tuple) and msg[0] == "info":
                self.info.set(msg[1])

        if not self.paused and self.popup is None and not self.calibrating and self.session_id is not None:
            deferred = self.scheduler.tick(self.inputs.last_key_ts)
            if deferred is not None:
                self._show_probe(deferred)

        self._refresh_status()
        self.root.after(1000, self._tick)

    def _refresh_status(self):
        el = int(time.time() - self.session_start)
        ctx = self.sampler.latest
        lines = [f"任务：{self.task}    已进行 {el // 3600}:{el % 3600 // 60:02d}:{el % 60:02d}"
                 + ("    【已暂停】" if self.paused else "")]
        cam = "暂停中（摄像头已关闭）" if self.paused and self.camera_ok else self.camera_status
        tr = self.tracker
        has_cam = tr is not None and self.camera_status == "运行中"
        if has_cam:
            cam += f"  {tr.fps:.0f} fps"
            if self.cal_info:
                cam += f"  校准误差 {self.cal_info['val_error_px']:.0f}px"
            if self.mapper is not None:
                cam += f"  点击校准 {self.mapper.implicit_accepted}"
        lines.append(f"摄像头：{cam}")
        if ctx and not self.paused:
            lines.append(f"前台：{ctx['exe']} · {CATEGORY_NAMES.get(ctx['category'], ctx['category'])}"
                         f" · {'任务内' if ctx['on_task'] else '任务外'}")
        if not self.paused:
            lines.append(f"下次自评：约 {self.scheduler.seconds_left() / 60:.0f} 分钟后")
        lines.append(f"今日自评：随机 {self.probe_counts.get('random', 0)} 次 · "
                     f"自察 {self.probe_counts.get('self', 0)} 次    已写入 {self.db.rows_written} 行")
        self.status.set("\n".join(lines))

        self.calib_btn.state(["!disabled"] if has_cam else ["disabled"])
        self.dot_chk.state(["!disabled"] if has_cam and self.mapper is not None else ["disabled"])
        self.warn.set(self.db.error or (posture_warning(self.mapper, tr.recent(1.0), self.cfg) if has_cam else ""))

    def _on_close(self):
        if messagebox.askyesno("退出", "退出并停止记录？", parent=self.root):
            self.quit()

    def quit(self):
        if self.popup:
            self.popup.win.destroy()
        self._end_session()
        self._stop_camera()
        self.sampler.stop()
        self.inputs.stop()
        self.hotkeys.stop()
        self.db.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
