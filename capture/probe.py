"""经验采样：随机时刻弹窗询问"刚才在干嘛"，以及自察走神热键。答案作为后续个人模型的标签。"""
import random
import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from storage.db import ANSWER_SKIPPED, ANSWER_TIMEOUT, ANSWERS

from . import screen


class ProbeScheduler:
    """纯逻辑：决定什么时候该弹窗。由界面线程每秒调用 tick()。"""

    def __init__(self, cfg: dict, rng: Optional[random.Random] = None):
        self.c = cfg["probes"]
        self.rng = rng or random.Random()
        self.next_due = 0.0
        self.deferred_since: Optional[float] = None
        self.reschedule()

    def reschedule(self, now: Optional[float] = None):
        now = time.monotonic() if now is None else now
        self.next_due = now + 60 * self.rng.uniform(self.c["min_minutes"], self.c["max_minutes"])
        self.deferred_since = None

    def tick(self, last_key_ts: float, now: Optional[float] = None) -> Optional[int]:
        """到点且不在打字（或已推迟够久）时返回推迟的毫秒数，表示应当弹窗；否则返回 None。"""
        now = time.monotonic() if now is None else now
        if now < self.next_due:
            return None
        typing = now - last_key_ts < 2.0
        if typing:
            if self.deferred_since is None:
                self.deferred_since = now
            if now - self.deferred_since < self.c["typing_defer_s"]:
                return None
        deferred = 0 if self.deferred_since is None else int((now - self.deferred_since) * 1000)
        return deferred

    def seconds_left(self) -> float:
        return max(0.0, self.next_due - time.monotonic())


class ProbePopup:
    """右下角的小弹窗，按数字键 1-5 或点击按钮作答。作答后把焦点还给原来的窗口。"""

    def __init__(self, root: tk.Tk, timeout_s: float, on_answer: Callable[[int, Optional[int]], None]):
        self.on_answer = on_answer
        self.prev_hwnd = (screen.foreground_window() or {}).get("hwnd")
        self.t0 = time.monotonic()
        self.done = False

        w = self.win = tk.Toplevel(root)
        w.title("注意力自评")
        w.attributes("-topmost", True)
        w.resizable(False, False)
        f = ttk.Frame(w, padding=12)
        f.pack()
        ttk.Label(f, text="弹窗出现前的那一刻，你在……", font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        for code, text in ANSWERS.items():
            ttk.Button(f, text=f"{code}  {text}", width=26,
                       command=lambda c=code: self._answer(c)).pack(fill="x", pady=2)
            w.bind(str(code), lambda e, c=code: self._answer(c))
        ttk.Label(f, text="Esc 跳过", foreground="#888").pack(anchor="e", pady=(4, 0))
        w.bind("<Escape>", lambda e: self._answer(ANSWER_SKIPPED))
        w.protocol("WM_DELETE_WINDOW", lambda: self._answer(ANSWER_SKIPPED))

        w.update_idletasks()
        sw, sh = screen.screen_size()
        w.geometry(f"+{sw - w.winfo_reqwidth() - 40}+{sh - w.winfo_reqheight() - 120}")
        w.lift()
        screen.force_foreground(w)
        w.focus_force()
        w.after(int(timeout_s * 1000), lambda: self._answer(ANSWER_TIMEOUT))

    def _answer(self, code: int):
        if self.done:
            return
        self.done = True
        response_ms = int((time.monotonic() - self.t0) * 1000) if code > 0 else None
        self.win.destroy()
        screen.set_foreground(self.prev_hwnd)
        self.on_answer(code, response_ms)


class Toast:
    """短暂的提示条（自察热键的确认），不抢焦点。"""

    def __init__(self, root: tk.Tk, text: str, ms: int = 1200):
        w = tk.Toplevel(root)
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        tk.Label(w, text=text, bg="#333", fg="white", padx=14, pady=8,
                 font=("Microsoft YaHei UI", 10)).pack()
        w.update_idletasks()
        sw, sh = screen.screen_size()
        w.geometry(f"+{sw - w.winfo_reqwidth() - 40}+{sh - w.winfo_reqheight() - 120}")
        screen.make_click_through(w)
        w.after(ms, w.destroy)
