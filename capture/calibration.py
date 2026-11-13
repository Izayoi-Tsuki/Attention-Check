"""视线映射模型与校准流程。

- GazeMapper：特征 → 屏幕坐标，二次多项式 + 岭回归；支持点击隐式校准持续修正漂移。
- CalibrationWindow：全屏 9 点校准 + 4 点验证（tkinter）。
"""
import threading
import time
import tkinter as tk
from collections import deque
from typing import Callable, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def _make_model(alpha: float):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


class GazeMapper:
    def __init__(self, cfg: dict):
        self.cfg_cal = cfg["calibration"]
        self.cfg_imp = cfg["implicit"]
        self.alpha = 1.0
        self._model = None
        self._X = np.empty((0, 0))
        self._Y = np.empty((0, 2))
        self._implicit: deque = deque(maxlen=self.cfg_imp["max_samples"])
        self._pending = 0
        self._lock = threading.Lock()
        self.implicit_accepted = 0
        self.implicit_rejected = 0
        self.pose_ref: Optional[np.ndarray] = None  # 校准时的平均头部姿态 [yaw, pitch, roll, tx, ty, tz]

    def __getstate__(self):  # 支持 pickle 存入数据库（锁不能序列化）
        d = self.__dict__.copy()
        d.pop("_lock")
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._lock = threading.Lock()

    def predict(self, feat: np.ndarray) -> np.ndarray:
        return self._model.predict(feat.reshape(1, -1))[0]

    def predict_many(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def fit_explicit(self, X: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> float:
        """留一点交叉验证选择正则强度，返回所选 alpha 下的交叉验证平均误差（像素）。"""
        best = None
        for alpha in self.cfg_cal["alphas"]:
            errs = []
            for g in np.unique(groups):
                tr, te = groups != g, groups == g
                m = _make_model(alpha).fit(X[tr], Y[tr])
                pred = np.median(m.predict(X[te]), axis=0)
                errs.append(np.linalg.norm(pred - Y[te][0]))
            err = float(np.mean(errs))
            if best is None or err < best[0]:
                best = (err, alpha)
        self.alpha = best[1]
        self._X, self._Y = X, Y
        self._model = _make_model(self.alpha).fit(X, Y)
        self.pose_ref = X[:, -6:].mean(axis=0)
        return best[0]

    def add_implicit(self, feat: np.ndarray, target) -> bool:
        target = np.asarray(target, dtype=float)
        if np.linalg.norm(self.predict(feat) - target) > self.cfg_imp["accept_radius_px"]:
            self.implicit_rejected += 1
            return False
        with self._lock:
            self._implicit.append((feat, target))
            self.implicit_accepted += 1
            self._pending += 1
            if self._pending < self.cfg_imp["refit_every"]:
                return True
            self._pending = 0
            fx, fy = zip(*self._implicit)
        X = np.vstack([self._X, np.array(fx)])
        Y = np.vstack([self._Y, np.array(fy)])
        w = np.r_[np.ones(len(self._X)), np.full(len(fx), float(self.cfg_imp["weight"]))]
        model = _make_model(self.alpha)
        model.fit(X, Y, ridge__sample_weight=w)
        self._model = model  # 引用替换是原子的，采集线程无需加锁
        return True


def posture_warning(mapper: Optional[GazeMapper], recent: list, cfg: dict) -> str:
    """与校准时的平均头部姿态相比，坐姿变化过大时返回提示文字。recent 为最近约 1 秒的帧。"""
    if mapper is None or mapper.pose_ref is None:
        return ""
    recent = [s for s in recent if s.face]
    if not recent:
        return ""
    cur = np.mean([[s.yaw, s.pitch, s.roll, *s.t] for s in recent], axis=0)
    ref = mapper.pose_ref
    p = cfg["posture"]
    if abs(cur[5] - ref[5]) > p["max_distance_change"] * abs(ref[5]):
        return "与屏幕的距离和校准时差别较大，建议重新校准"
    if np.hypot(cur[3] - ref[3], cur[4] - ref[4]) > p["max_shift_cm"]:
        return "头部位置和校准时偏移较大，建议重新校准"
    if abs(cur[0] - ref[0]) > p["max_yaw_deg"]:
        return "头部转向和校准时差别较大，建议重新校准"
    return ""


class CalibrationWindow:
    """全屏校准。完成后调用 on_done(mapper, info)；取消时 on_done(None, None)。"""

    def __init__(self, root: tk.Tk, tracker, cfg: dict, screen: tuple,
                 on_done: Callable[[Optional[GazeMapper], Optional[dict]], None]):
        self.root, self.tracker, self.cfg = root, tracker, cfg
        self.c = cfg["calibration"]
        self.sw, self.sh = screen
        self.on_done = on_done
        self._closed = False

        m = self.c["margin"]
        xs = [m, 0.5, 1 - m]
        self.points = [(x, y) for y in xs for x in xs]
        self.val_points = [tuple(p) for p in self.c["validation_points"]]

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.sw}x{self.sh}+0+0")
        self.win.attributes("-topmost", True)
        self.cv = tk.Canvas(self.win, bg=self.c["background"], highlightthickness=0, cursor="none")
        self.cv.pack(fill="both", expand=True)
        self.win.bind("<Escape>", lambda e: self._finish(None, None))
        self.win.focus_force()
        self._intro()

    # ---------- 各阶段 ----------
    def _intro(self):
        self.cv.delete("all")
        self.cv.configure(cursor="arrow")
        self._text(self.sw / 2, self.sh / 2 - 60, "视线校准", 28)
        self._text(self.sw / 2, self.sh / 2 + 10,
                   "以平常的坐姿坐好，头部尽量不动，只用眼睛注视依次出现的圆点中心。\n"
                   "共 9 个校准点 + 4 个验证点，约 25 秒。", 14)
        self._text(self.sw / 2, self.sh / 2 + 90, "按 空格 开始    Esc 取消", 14)
        self.win.bind("<space>", lambda e: self._start())

    def _start(self):
        self.win.unbind("<space>")
        self.cv.configure(cursor="none")
        self.X, self.Y, self.G = [], [], []
        self.val = []  # (target, frames)
        self._run_points(self.points, self._collect_cal, self._after_cal)

    def _run_points(self, points, collect, done):
        queue = list(enumerate(points))
        retries = {}

        def step():
            if self._closed:
                return
            if not queue:
                done()
                return
            idx, (fx, fy) = queue[0]
            target = (fx * self.sw, fy * self.sh)
            self._animate_point(target, lambda frames: after_point(idx, target, frames))

        def after_point(idx, target, frames):
            if len(frames) < self.c["min_samples"]:
                retries[idx] = retries.get(idx, 0) + 1
                if retries[idx] > self.c["max_retries"]:
                    self._fail("有效帧太少：请确认摄像头能清楚看到你的脸、光线充足。")
                    return
            else:
                collect(idx, target, frames)
                queue.pop(0)
            self.win.after(250, step)

        step()

    def _animate_point(self, target, on_frames):
        dur, settle = self.c["point_seconds"], self.c["settle_seconds"]
        t_start = time.monotonic()
        x, y = target

        def tick():
            if self._closed:
                return
            el = time.monotonic() - t_start
            self.cv.delete("all")
            if el >= dur:
                frames = self.tracker.samples_between(t_start + settle, t_start + dur)
                on_frames(frames)
                return
            r = 30 - 20 * min(el / settle, 1.0)
            self.cv.create_oval(x - r, y - r, x + r, y + r, outline="white", width=3)
            self.cv.create_oval(x - 4, y - 4, x + 4, y + 4, fill="black", outline="")
            self.win.after(16, tick)

        tick()

    def _collect_cal(self, idx, target, frames):
        for f in frames:
            self.X.append(f.feat)
            self.Y.append(target)
            self.G.append(idx)

    def _collect_val(self, idx, target, frames):
        self.val.append((target, frames))

    def _after_cal(self):
        self.mapper = GazeMapper(self.cfg)
        self.cv_err = self.mapper.fit_explicit(np.array(self.X), np.array(self.Y), np.array(self.G))
        self._run_points(self.val_points, self._collect_val, self._after_val)

    def _after_val(self):
        results = []
        for target, frames in self.val:
            pred = np.median(self.mapper.predict_many(np.array([f.feat for f in frames])), axis=0)
            results.append((np.array(target), pred))
        errors = [float(np.linalg.norm(p - t)) for t, p in results]
        self.info = dict(
            val_error_px=float(np.mean(errors)),
            cv_error_px=self.cv_err,
            alpha=self.mapper.alpha,
            n_frames=len(self.X),
        )
        # 验证点也并入训练集，得到最终模型
        n_cal = len(self.points)
        for i, (target, frames) in enumerate(self.val):
            for f in frames:
                self.X.append(f.feat)
                self.Y.append(target)
                self.G.append(n_cal + i)
        self.mapper.fit_explicit(np.array(self.X), np.array(self.Y), np.array(self.G))
        self._show_result(results, errors)

    def _show_result(self, results, errors):
        self.cv.delete("all")
        self.cv.configure(cursor="arrow")
        for (t, p), e in zip(results, errors):
            self.cv.create_line(*t, *p, fill="yellow", width=2)
            self.cv.create_oval(t[0] - 6, t[1] - 6, t[0] + 6, t[1] + 6, outline="white", width=2)
            self.cv.create_oval(p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6, fill="#00e000", outline="")
            self._text(t[0], t[1] + 24, f"{e:.0f}px", 11)
        err = self.info["val_error_px"]
        verdict = "良好" if err < 120 else ("可用" if err < 200 else "偏差较大，建议重做")
        self._text(self.sw / 2, self.sh / 2 - 30,
                   f"验证平均误差：{err:.0f} px（屏幕宽度的 {err / self.sw * 100:.1f}%）  —  {verdict}", 18)
        self._text(self.sw / 2, self.sh / 2 + 20,
                   "白圈 = 目标点，绿点 = 估计的视线位置", 12)
        self._text(self.sw / 2, self.sh / 2 + 70, "空格 接受    R 重新校准    Esc 取消", 14)
        self.win.bind("<space>", lambda e: self._finish(self.mapper, self.info))
        self.win.bind("<r>", lambda e: self._restart())
        self.win.bind("<R>", lambda e: self._restart())

    def _restart(self):
        for k in ("<r>", "<R>", "<space>"):
            self.win.unbind(k)
        self._intro()

    def _fail(self, msg):
        self.cv.delete("all")
        self.cv.configure(cursor="arrow")
        self._text(self.sw / 2, self.sh / 2 - 20, msg, 16)
        self._text(self.sw / 2, self.sh / 2 + 40, "按 空格 重试    Esc 取消", 14)
        self.win.bind("<space>", lambda e: self._start())

    def _finish(self, mapper, info):
        self._closed = True
        self.win.destroy()
        self.on_done(mapper, info)

    def _text(self, x, y, s, size):
        self.cv.create_text(x, y, text=s, fill="black", font=("Microsoft YaHei UI", size), justify="center")
