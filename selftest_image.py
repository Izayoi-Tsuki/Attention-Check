"""用一张静态人脸图片验证 MediaPipe 特征提取（没有摄像头时使用）。

运行：  .venv\\Scripts\\python.exe selftest_image.py <图片路径>
"""
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from capture.gaze import FEATURE_NAMES, ensure_model, extract
from settings import load_config

cfg = load_config()
path = ensure_model(cfg["model"]["path"], cfg["model"]["url"])
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=path),
    running_mode=vision.RunningMode.IMAGE,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)
img = cv2.imread(sys.argv[1])
rgb = cv2.cvtColor(cv2.resize(img, (640, 480)), cv2.COLOR_BGR2RGB)
with vision.FaceLandmarker.create_from_options(options) as lm:
    t = time.perf_counter()
    result = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    ms = (time.perf_counter() - t) * 1000
s = extract(result, 640, 480, 0.0, cfg["gaze"]["blink_threshold"])
print(f"推理耗时 {ms:.0f}ms  人脸={s.face}  有效={s.valid}  眨眼 L/R={s.blink_l:.2f}/{s.blink_r:.2f}")
if s.face:
    for name, v in zip(FEATURE_NAMES, s.feat):
        print(f"  {name:18s} {v:+9.3f}")
