"""屏幕上下文采集：前台窗口、应用分类、键鼠活跃度（只计数，不记录内容）。"""
import hashlib
import threading
import time
from typing import Callable, Optional

from pynput import keyboard, mouse

from . import screen


class Classifier:
    """按 config.yaml 中 categories 规则给窗口分类，并判断是否属于当前声明的任务。"""

    def __init__(self, rules: list):
        self.rules = []
        for r in rules:
            self.rules.append(dict(
                category=r["category"],
                exe={e.lower() for e in r.get("exe", [])},
                title=[t.lower() for t in r.get("title", [])],
            ))
        self.task_terms: list = []

    def set_task_apps(self, task_apps: str):
        """task_apps：逗号分隔的 exe 名或标题关键词，如 "Code.exe, 论文"。为空时以 task 类别判定。"""
        self.task_terms = [t.strip().lower() for t in task_apps.replace("，", ",").split(",") if t.strip()]

    def classify(self, exe: str, title: str) -> tuple:
        """返回 (类别, 命中的规则关键词)。关键词用于报告中聚合干扰源，比如把各个 B 站页面合并为 "bilibili"。"""
        exe_l, title_l = exe.lower(), title.lower()
        for r in self.rules:
            for t in r["title"]:
                if t in title_l:
                    return r["category"], t
            if exe_l in r["exe"]:
                return r["category"], exe
        return "neutral", exe

    def on_task(self, exe: str, title: str, category: str) -> bool:
        if not self.task_terms:
            return category == "task"
        exe_l, title_l = exe.lower(), title.lower()
        return any(t == exe_l or t in title_l for t in self.task_terms)


class InputMonitor:
    """全局键鼠监听，只累计次数。点击事件可额外转发（用于视线的点击隐式校准）。"""

    def __init__(self, on_click: Optional[Callable] = None):
        self._lock = threading.Lock()
        self._counts = dict(keys=0, clicks=0, mouse_moves=0, scrolls=0)
        self.last_key_ts = 0.0
        self._on_click_cb = on_click
        self._mouse = mouse.Listener(on_move=self._move, on_click=self._click, on_scroll=self._scroll)
        self._kb = keyboard.Listener(on_press=self._press)

    def start(self):
        self._mouse.start()
        self._kb.start()

    def stop(self):
        self._mouse.stop()
        self._kb.stop()

    def take_counts(self) -> dict:
        with self._lock:
            c = self._counts
            self._counts = dict(keys=0, clicks=0, mouse_moves=0, scrolls=0)
        return c

    def _inc(self, k):
        with self._lock:
            self._counts[k] += 1

    def _move(self, x, y):
        self._inc("mouse_moves")

    def _scroll(self, x, y, dx, dy):
        self._inc("scrolls")

    def _press(self, key):
        self.last_key_ts = time.monotonic()
        self._inc("keys")

    def _click(self, x, y, button, pressed):
        if pressed:
            self._inc("clicks")
            if self._on_click_cb:
                self._on_click_cb(x, y)


def _privacy_title(title: str, mode: str) -> str:
    if mode == "drop":
        return ""
    if mode == "hash":
        return hashlib.sha1(title.encode("utf-8")).hexdigest()[:12] if title else ""
    return title


class ContextSampler(threading.Thread):
    """每秒采样一次前台窗口与键鼠计数，生成 context 表的一行，交给 sink 写库。"""

    def __init__(self, cfg: dict, inputs: InputMonitor, classifier: Classifier, sink: Callable[[dict], None]):
        super().__init__(daemon=True)
        self.interval = cfg["context"]["sample_seconds"]
        self.title_mode = cfg["privacy"]["titles"]
        self.inputs = inputs
        self.classifier = classifier
        self.sink = sink
        self.paused = False
        self.latest: Optional[dict] = None
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        next_t = time.monotonic()
        while not self._stop_event.is_set():
            next_t += self.interval
            self._stop_event.wait(max(0.0, next_t - time.monotonic()))
            counts = self.inputs.take_counts()  # 暂停期间也清空，避免恢复后第一秒计数虚高
            if self.paused:
                continue
            fg = screen.foreground_window() or dict(exe="", title="", cls="", rect=(0, 0, 0, 0))
            category, key = self.classifier.classify(fg["exe"], fg["title"])
            row = dict(
                ts=time.time(),
                exe=fg["exe"],
                title=_privacy_title(fg["title"], self.title_mode),
                category=category,
                on_task=int(self.classifier.on_task(fg["exe"], fg["title"], category)),
                rect=fg["rect"],
                idle_s=screen.idle_seconds(),
                key=key,
                **counts,
            )
            self.latest = row
            self.sink(row)
