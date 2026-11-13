"""无界面自检：映射模型、滤波器、窗口查询、摄像头特征提取。

运行：  .venv\\Scripts\\python.exe selftest.py [--no-camera]
"""
import sys
import time

import numpy as np

from capture import screen
from capture.calibration import GazeMapper
from capture.filters import OneEuroFilter
from capture.gaze import FEATURE_NAMES, GazeTracker
from settings import load_config


def test_mapper(cfg):
    rng = np.random.default_rng(0)
    n_feat = len(FEATURE_NAMES)
    w_true = rng.normal(size=(n_feat, 2)) * 200
    targets = [(x, y) for y in (150, 540, 930) for x in (150, 960, 1770)]
    X, Y, G = [], [], []
    for g, t in enumerate(targets):
        base = np.linalg.lstsq(w_true.T, np.array(t) - 960, rcond=None)[0]
        for _ in range(15):
            X.append(base + rng.normal(scale=0.01, size=n_feat))
            Y.append(t)
            G.append(g)
    m = GazeMapper(cfg)
    cv_err = m.fit_explicit(np.array(X), np.array(Y), np.array(G))
    err = np.linalg.norm(m.predict(X[0]) - np.array(Y[0]))
    assert err < 50, err
    accepted = m.add_implicit(X[20], Y[20])
    rejected = m.add_implicit(X[20], (Y[20][0] + 2000, Y[20][1]))
    assert accepted and not rejected
    print(f"[ok] 映射模型  alpha={m.alpha}  交叉验证误差={cv_err:.1f}px  训练点误差={err:.1f}px")


def test_filter():
    f = OneEuroFilter(0.8, 0.003, 1.0)
    rng = np.random.default_rng(1)
    out = [f(np.array([500, 500]) + rng.normal(scale=30, size=2), i / 15) for i in range(60)]
    jitter_in, jitter_out = 30, np.std(np.array(out[20:]), axis=0).mean()
    assert jitter_out < jitter_in
    print(f"[ok] One Euro 滤波  抖动 {jitter_in}px → {jitter_out:.1f}px")


def test_window_probe():
    sw, sh = screen.screen_size()
    print(f"[ok] 屏幕尺寸 {sw}x{sh}")
    for fx, fy in ((0.5, 0.5), (0.1, 0.1), (0.9, 0.5), (0.5, 0.99)):
        info = screen.window_at(fx * sw, fy * sh)
        print(f"     ({fx:.2f},{fy:.2f}) → {info and (info['exe'], info['title'][:40], info['cls'])}")


def test_camera(cfg):
    tr = GazeTracker(cfg)
    tr.start()
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6 and tr.error is None:
        time.sleep(0.2)
    tr.stop()
    tr.join(timeout=3)
    if tr.error:
        print(f"[失败] 摄像头：{tr.error}")
        return
    frames = tr.recent(10)
    faces = [s for s in frames if s.face]
    valid = [s for s in faces if s.valid]
    print(f"[ok] 摄像头  fps={tr.fps:.1f}  帧={len(frames)}  有人脸={len(faces)}  有效={len(valid)}")
    if valid:
        f = np.array([s.feat for s in valid])
        for name, mu, sd in zip(FEATURE_NAMES, f.mean(0), f.std(0)):
            print(f"     {name:18s} {mu:+9.3f} ± {sd:.3f}")


if __name__ == "__main__":
    screen.set_dpi_awareness()
    cfg = load_config()
    test_mapper(cfg)
    test_filter()
    test_window_probe()
    if "--no-camera" not in sys.argv:
        test_camera(cfg)
