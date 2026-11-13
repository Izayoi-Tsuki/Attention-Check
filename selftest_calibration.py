"""用模拟数据快速走一遍校准界面流程（不需要摄像头），只验证代码路径。"""
import tkinter as tk

import numpy as np

from capture import screen
from capture.calibration import CalibrationWindow
from capture.gaze import FEATURE_NAMES, FrameSample
from settings import load_config


class FakeTracker:
    def __init__(self):
        self.rng = np.random.default_rng(0)

    def samples_between(self, t0, t1, valid_only=True):
        return [FrameSample(ts=t0, face=True, valid=True, feat=self.rng.normal(size=len(FEATURE_NAMES)))
                for _ in range(10)]


screen.set_dpi_awareness()
cfg = load_config()
cfg["calibration"].update(point_seconds=0.3, settle_seconds=0.1)
root = tk.Tk()
root.withdraw()
result = {}


def done(mapper, info):
    result.update(mapper=mapper, info=info)
    root.destroy()


cal = CalibrationWindow(root, FakeTracker(), cfg, screen.screen_size(), done)
root.after(300, lambda: cal.win.event_generate("<space>"))


def accept_when_ready():
    if hasattr(cal, "info"):
        cal.win.event_generate("<space>")
    else:
        root.after(200, accept_when_ready)


root.after(1000, accept_when_ready)
root.mainloop()
m = result["mapper"]
print("info:", result["info"])
print("predict:", m.predict(np.zeros(len(FEATURE_NAMES))), "pose_ref:", m.pose_ref.shape)
