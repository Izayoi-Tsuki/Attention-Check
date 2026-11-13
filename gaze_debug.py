"""第 1 步调试程序：视线采集 + 校准 + 实时视线点叠加。

运行：  .venv\\Scripts\\python.exe gaze_debug.py
"""
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
from pynput import mouse

from capture import screen
from capture.calibration import CalibrationWindow, posture_warning
from capture.gaze import GazeTracker
from capture.overlay import GazeOverlay
from settings import load_config


class App:
    def __init__(self):
        screen.set_dpi_awareness()
        self.cfg = load_config()
        self.sw, self.sh = screen.screen_size()

        self.root = tk.Tk()
        self.root.title("注意力追踪 · 视线调试")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._build_panel()

        self.tracker = GazeTracker(self.cfg)
        self.tracker.start()
        self.overlay = GazeOverlay(self.root, self.sw, self.sh, self.cfg["overlay"]["dot_radius"])
        self.overlay.set_visible(False)

        self.calibrating = False
        self.cal_info = None
        self.win_info = None
        self._last_probe = 0.0
        self._photo = None

        self.listener = mouse.Listener(on_click=self._on_click)
        self.listener.start()

        self.root.after(30, self._tick)
        self.root.after(1500, self._wait_camera_then_calibrate)

    # ---------- 界面 ----------
    def _build_panel(self):
        f = ttk.Frame(self.root, padding=8)
        f.pack(fill="both", expand=True)
        self.preview = ttk.Label(f)
        self.preview.pack()
        self.stats = tk.StringVar(value="正在启动摄像头和模型……")
        ttk.Label(f, textvariable=self.stats, font=("Consolas", 10), justify="left",
                  wraplength=460).pack(anchor="w", pady=6)
        self.warn = tk.StringVar()
        ttk.Label(f, textvariable=self.warn, foreground="#c00000").pack(anchor="w")

        bar = ttk.Frame(f)
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="重新校准", command=self.calibrate).pack(side="left")
        self.show_dot = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="显示视线点", variable=self.show_dot,
                        command=self._sync_overlay).pack(side="left", padx=8)
        self.implicit_on = tk.BooleanVar(value=self.cfg["implicit"]["enabled"])
        self.implicit_enabled = self.implicit_on.get()  # 供鼠标线程读取，tkinter 变量不能跨线程访问
        ttk.Checkbutton(bar, text="点击隐式校准", variable=self.implicit_on,
                        command=lambda: setattr(self, "implicit_enabled", self.implicit_on.get())).pack(side="left")
        ttk.Button(bar, text="退出", command=self.quit).pack(side="right")

        self.root.update_idletasks()
        self.root.geometry(f"+{self.sw - self.root.winfo_reqwidth() - 380}+40")

    def _sync_overlay(self):
        self.overlay.set_visible(self.show_dot.get() and self.tracker.mapper is not None and not self.calibrating)

    # ---------- 校准 ----------
    def _wait_camera_then_calibrate(self):
        if self.tracker.error:
            return
        if self.tracker.latest is None:
            self.root.after(500, self._wait_camera_then_calibrate)
            return
        self.calibrate()

    def calibrate(self):
        if self.calibrating or self.tracker.latest is None:
            return
        self.calibrating = True
        self._sync_overlay()
        self.root.withdraw()
        CalibrationWindow(self.root, self.tracker, self.cfg, (self.sw, self.sh), self._on_calibrated)

    def _on_calibrated(self, mapper, info):
        self.calibrating = False
        if mapper is not None:
            self.tracker.mapper = mapper
            self.cal_info = info
        self.root.deiconify()
        self._sync_overlay()

    def _on_click(self, x, y, button, pressed):
        # pynput 回调线程：只读帧缓冲、更新映射模型，不碰 tkinter
        mapper = self.tracker.mapper
        if not pressed or self.calibrating or mapper is None or not self.implicit_enabled:
            return
        t = time.monotonic()
        w0, w1 = self.cfg["implicit"]["window"]
        frames = self.tracker.samples_between(t - w0, t - w1)
        if len(frames) >= 2:
            mapper.add_implicit(np.median([f.feat for f in frames], axis=0), (x, y))

    # ---------- 刷新 ----------
    def _tick(self):
        tr = self.tracker
        s = tr.latest
        if tr.preview is not None:
            img = Image.fromarray(tr.preview).resize((320, 240))
            self._photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self._photo)

        if tr.error:
            self.stats.set(f"错误：{tr.error}")
        elif s is not None:
            gaze = s.gaze
            now = time.monotonic()
            if gaze is not None and now - self._last_probe > self.cfg["overlay"]["window_probe_ms"] / 1000:
                self._last_probe = now
                self.win_info = screen.window_at(*gaze)
            on_screen = gaze is not None and 0 <= gaze[0] < self.sw and 0 <= gaze[1] < self.sh
            if gaze is None:
                where = "—"
            elif not on_screen:
                where = "屏幕外"
            elif self.win_info:
                where = f"{self.win_info['exe']} | {self.win_info['title'][:40]}"
            else:
                where = "?"

            mapper = tr.mapper
            lines = [
                f"FPS        {tr.fps:5.1f}      人脸 {'是' if s.face else '否'}",
                f"眨眼 L/R   {s.blink_l:.2f} / {s.blink_r:.2f}",
                f"头部 yaw/pitch/roll  {s.yaw:+5.1f} {s.pitch:+5.1f} {s.roll:+5.1f}",
                f"头部位置 cm  {'—' if s.t is None else ' '.join(f'{v:+.1f}' for v in s.t)}",
                f"视线 px    {'—' if gaze is None else f'{gaze[0]:.0f}, {gaze[1]:.0f}'}",
                f"注视窗口   {where}",
            ]
            if self.cal_info:
                lines.append(f"校准误差   验证 {self.cal_info['val_error_px']:.0f}px  "
                             f"交叉验证 {self.cal_info['cv_error_px']:.0f}px")
            if mapper is not None:
                lines.append(f"点击校准   接受 {mapper.implicit_accepted}  拒绝 {mapper.implicit_rejected}")
            else:
                lines.append("尚未校准")
            self.stats.set("\n".join(lines))
            self.warn.set(posture_warning(mapper, tr.recent(1.0), self.cfg))

            if self.overlay.visible:
                xy = None
                if gaze is not None:
                    xy = (min(max(gaze[0], 0), self.sw - 1), min(max(gaze[1], 0), self.sh - 1))
                self.overlay.update(xy, where)
        self.root.after(30, self._tick)

    def quit(self):
        self.tracker.stop()
        self.listener.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
