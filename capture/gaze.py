"""摄像头 + MediaPipe Face Landmarker：逐帧提取视线特征，并（校准后）映射为屏幕坐标。

不保存任何图像，只保留由关键点计算出的数值特征。
"""
import math
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .filters import OneEuroFilter

# MediaPipe 478 点人脸网格中的眼部索引。
# "右眼"指本人右眼（画面左侧）；每只眼的两个眼角按"画面左 → 画面右"排列，
# 使两只眼的水平特征方向一致。
RIGHT_EYE = dict(a=33, b=133, iris=468, up=159, lo=145)
LEFT_EYE = dict(a=362, b=263, iris=473, up=386, lo=374)

LOOK_BLENDSHAPES = [
    "eyeLookInLeft", "eyeLookOutLeft", "eyeLookUpLeft", "eyeLookDownLeft",
    "eyeLookInRight", "eyeLookOutRight", "eyeLookUpRight", "eyeLookDownRight",
]

FEATURE_NAMES = (
    ["u_r", "v_r", "open_r", "u_l", "v_l", "open_l"]
    + LOOK_BLENDSHAPES
    + ["yaw", "pitch", "roll", "tx", "ty", "tz"]
)


@dataclass
class FrameSample:
    ts: float                       # time.monotonic() 秒
    face: bool                      # 是否检测到人脸
    valid: bool                     # 视线特征是否可用（有人脸且未眨眼）
    feat: Optional[np.ndarray]      # 视线特征向量，见 FEATURE_NAMES
    blink_l: float = 0.0
    blink_r: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    t: Optional[np.ndarray] = None  # 头部平移 (x, y, z)，单位约为 cm
    gaze: Optional[np.ndarray] = None      # 平滑后的屏幕坐标（像素）
    gaze_raw: Optional[np.ndarray] = None  # 未平滑的屏幕坐标


def ensure_model(path: str, url: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    print(f"首次运行，正在下载模型到 {p} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(p)
    return str(p)


def _rotation_to_euler(r: np.ndarray):
    """旋转矩阵 → (yaw, pitch, roll)，单位度。"""
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def _eye_features(pts: np.ndarray, eye: dict):
    """虹膜在眼角连线上的相对位置 u、垂直偏移 v、睁眼程度 open（均按眼宽归一化）。"""
    a, b, c = pts[eye["a"]], pts[eye["b"]], pts[eye["iris"]]
    axis = b - a
    width = float(np.linalg.norm(axis)) + 1e-6
    ax = axis / width
    perp = np.array([-ax[1], ax[0]])  # 图像坐标 y 向下，perp 指向下方
    d = c - a
    u = float(d @ ax) / width
    v = float(d @ perp) / width
    opening = float(np.linalg.norm(pts[eye["up"]] - pts[eye["lo"]])) / width
    return u, v, opening


def extract(result, w: int, h: int, ts: float, blink_threshold: float) -> FrameSample:
    if not result.face_landmarks:
        return FrameSample(ts=ts, face=False, valid=False, feat=None)

    lms = result.face_landmarks[0]
    pts = np.array([(p.x * w, p.y * h) for p in lms], dtype=float)
    bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
    m = np.array(result.facial_transformation_matrixes[0])
    yaw, pitch, roll = _rotation_to_euler(m[:3, :3])
    t = m[:3, 3].copy()

    feat = np.array(
        [*_eye_features(pts, RIGHT_EYE), *_eye_features(pts, LEFT_EYE)]
        + [bs.get(k, 0.0) for k in LOOK_BLENDSHAPES]
        + [yaw, pitch, roll, *t],
        dtype=float,
    )
    blink_l, blink_r = bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)
    valid = max(blink_l, blink_r) < blink_threshold
    return FrameSample(
        ts=ts, face=True, valid=valid, feat=feat,
        blink_l=blink_l, blink_r=blink_r, yaw=yaw, pitch=pitch, roll=roll, t=t,
    )


def _draw_preview(frame: np.ndarray, result) -> np.ndarray:
    if result is not None and result.face_landmarks:
        h, w = frame.shape[:2]
        lms = result.face_landmarks[0]
        for eye in (RIGHT_EYE, LEFT_EYE):
            for key, color, r in (("a", (0, 200, 255), 2), ("b", (0, 200, 255), 2),
                                  ("up", (255, 200, 0), 2), ("lo", (255, 200, 0), 2),
                                  ("iris", (0, 255, 0), 3)):
                p = lms[eye[key]]
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), r, color, -1)
    return cv2.flip(frame, 1)  # 镜像显示，更符合照镜子的直觉


class GazeTracker(threading.Thread):
    """后台线程：持续读取摄像头、提取特征、维护最近若干秒的帧缓冲。

    设置 `mapper` 属性（见 calibration.GazeMapper）后，每帧同时输出屏幕视线坐标。
    """

    def __init__(self, cfg: dict):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.mapper = None
        self.on_sample = None  # 可选回调 on_sample(FrameSample)，在采集线程中调用（用于写库）
        self.error: Optional[str] = None
        self.fps = 0.0
        self.preview: Optional[np.ndarray] = None  # 最新预览帧（RGB，已镜像）
        self.latest: Optional[FrameSample] = None

        cam = cfg["camera"]
        self._interval = 1.0 / cam["target_fps"]
        self._buf: deque[FrameSample] = deque(maxlen=int(cam["target_fps"] * cfg["gaze"]["buffer_seconds"]))
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        s = cfg["smoothing"]
        self._filter = OneEuroFilter(s["min_cutoff"], s["beta"], s["d_cutoff"])
        self._last_face_ts = 0.0
        self._last_gaze: Optional[np.ndarray] = None

    def stop(self):
        self._stop_event.set()

    def samples_between(self, t0: float, t1: float, valid_only: bool = True) -> list:
        with self._lock:
            return [s for s in self._buf if t0 <= s.ts <= t1 and (s.valid or not valid_only)]

    def recent(self, seconds: float) -> list:
        now = time.monotonic()
        return self.samples_between(now - seconds, now, valid_only=False)

    def run(self):
        try:
            self._run()
        except Exception as e:  # 让界面能显示错误，而不是线程静默退出
            self.error = f"{type(e).__name__}: {e}"
            raise

    def _run(self):
        cam = self.cfg["camera"]
        model_path = ensure_model(self.cfg["model"]["path"], self.cfg["model"]["url"])
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        cap = cv2.VideoCapture(cam["index"], cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
        if not cap.isOpened():
            self.error = (f"无法打开摄像头 index={cam['index']}：请确认摄像头已连接、未被其他程序占用；"
                          "有多个摄像头时可修改 config.yaml 中的 camera.index")
            return

        blink_th = self.cfg["gaze"]["blink_threshold"]
        t0 = time.monotonic()
        last_proc = 0.0
        last_ms = -1
        fps_count, fps_t = 0, t0
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                ts = time.monotonic()
                if not ok:
                    time.sleep(0.01)
                    continue
                if ts - last_proc < self._interval:
                    continue
                last_proc = ts

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ts_ms = max(int((ts - t0) * 1000), last_ms + 1)  # VIDEO 模式要求时间戳严格递增
                last_ms = ts_ms
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)

                h, w = frame.shape[:2]
                sample = extract(result, w, h, ts, blink_th)
                self._map(sample)
                with self._lock:
                    self._buf.append(sample)
                self.latest = sample
                if self.on_sample:
                    self.on_sample(sample)
                self.preview = _draw_preview(rgb, result)

                fps_count += 1
                if ts - fps_t >= 1.0:
                    self.fps = fps_count / (ts - fps_t)
                    fps_count, fps_t = 0, ts
        cap.release()

    def _map(self, s: FrameSample):
        mapper = self.mapper
        if s.face:
            self._last_face_ts = s.ts
        elif s.ts - self._last_face_ts > 0.5:
            self._filter.reset()
            self._last_gaze = None
        if mapper is None:
            return
        if s.valid:
            raw = mapper.predict(s.feat)
            s.gaze_raw = raw
            self._last_gaze = self._filter(raw, s.ts)
        # 眨眼帧沿用上一帧视线；人脸丢失超过 0.5s 后视线为空
        s.gaze = None if self._last_gaze is None else self._last_gaze.copy()
