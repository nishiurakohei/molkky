"""
molkky_pose_pipeline.py  v2.0
==============================
モルック投擲動画から骨格キーポイントを検知し、
時系列 CSV と可視化動画を出力するパイプライン。

対応フォーマット:
    スマートフォン縦動画 (1080x1920) を主対象。
    横動画・アクションカムにも対応（自動判定）。

依存ライブラリ:
    pip install mediapipe opencv-python scipy numpy

モデルダウンロード（初回のみ・約30MB）:
    python molkky_pose_pipeline.py --download-model

基本的な使い方:
    python molkky_pose_pipeline.py --input throw.mp4
    python molkky_pose_pipeline.py --input throw.mp4 --output-dir ./results
    python molkky_pose_pipeline.py --input throw.mp4 --no-video  # CSVのみ

GPU/CPU:
    CUDA が使用可能な場合は自動で GPU を選択。
    強制 CPU: --device cpu
    強制 GPU: --device gpu

縦動画の注意:
    EXIF の回転情報が埋め込まれた MP4 は OpenCV が無視するため、
    --rotate 90 / 270 で手動補正してください（通常は不要）。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

# -- MediaPipe Tasks API (mediapipe >= 0.10) -----------------------------------
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# =============================================================================
# 定数・デフォルト設定
# =============================================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)
MODEL_PATH = Path("pose_landmarker_heavy.task")

# 背面・斜めアングルで主要な 8 キーポイント（MediaPipe インデックス）
TARGET_KP: dict[str, int] = {
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_wrist":     15,
    "right_wrist":    16,
}

# 信頼度閾値
VISIBILITY_THRESHOLD = 0.60

# リリース検出パラメータ
RELEASE_PEAK_MIN_HEIGHT   = 0.012   # 正規化座標/秒の手首速度下限
RELEASE_PEAK_MIN_DISTANCE = 5       # フレーム数（連続検出抑制）
RELEASE_WINDOW_SEC        = 0.5     # リリース前後のハイライト幅 [s]

# 処理解像度：長辺をこの値に縮小して推定（速度と精度のバランス）
PROC_LONG_SIDE = 960

# 描画色 (BGR)
COLOR_KP      = (0,  230, 120)
COLOR_BONE    = (100, 180, 255)
COLOR_RELEASE = (0,   60, 255)
COLOR_WINDOW  = (0,  200, 255)

# =============================================================================
# データクラス
# =============================================================================

@dataclass
class VideoMeta:
    """動画メタ情報"""
    width:        int
    height:       int
    fps:          float
    total_frames: int
    is_portrait:  bool   # True = 縦動画
    proc_width:   int    # 推定処理用リサイズ後の幅
    proc_height:  int    # 推定処理用リサイズ後の高さ

    @classmethod
    def from_cap(cls, cap: cv2.VideoCapture) -> "VideoMeta":
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        portrait = h > w
        scale = min(1.0, PROC_LONG_SIDE / max(w, h))
        pw = int(w * scale)
        ph = int(h * scale)
        pw = pw if pw % 2 == 0 else pw - 1
        ph = ph if ph % 2 == 0 else ph - 1
        return cls(w, h, fps, total, portrait, pw, ph)

    def __str__(self) -> str:
        orient = "縦" if self.is_portrait else "横"
        return (
            f"{self.width}x{self.height} ({orient})  fps={self.fps:.1f}"
            f"  総フレーム={self.total_frames}"
            f"  処理解像度={self.proc_width}x{self.proc_height}"
        )


@dataclass
class KeypointRecord:
    """1 フレームの 1 キーポイント"""
    frame:        int
    time_sec:     float
    name:         str
    x_norm:       float    # 正規化座標 [0,1]
    y_norm:       float
    x_px:         int      # ピクセル座標（元解像度）
    y_px:         int
    visibility:   float
    valid:        bool
    release_flag: int = 0
    in_window:    int = 0


@dataclass
class FrameResult:
    """1 フレームの全キーポイント"""
    frame:     int
    time_sec:  float
    fps:       float
    keypoints: list[KeypointRecord] = field(default_factory=list)


# =============================================================================
# デバイス検出
# =============================================================================

def detect_device(prefer: str = "auto") -> str:
    if prefer in ("cpu", "gpu"):
        return prefer
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def build_pose_landmarker(
    model_path: Path,
    device: str = "cpu",
    mode: str = "VIDEO",
) -> PoseLandmarker:
    running_mode = {
        "IMAGE": VisionTaskRunningMode.IMAGE,
        "VIDEO": VisionTaskRunningMode.VIDEO,
    }[mode]

    base_kwargs: dict = {"model_asset_path": str(model_path)}
    if device == "gpu":
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions as BO
            base_kwargs["delegate"] = BO.Delegate.GPU
        except Exception:
            print("  [警告] GPU デリゲート設定失敗。CPU にフォールバック。")

    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(**base_kwargs),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.50,
        min_pose_presence_confidence=0.50,
        min_tracking_confidence=0.50,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(opts)


# =============================================================================
# フレーム前処理
# =============================================================================

def preprocess_frame(
    bgr: np.ndarray,
    meta: VideoMeta,
    rotate_code: Optional[int] = None,
) -> np.ndarray:
    """
    砂地・縦動画向け前処理。
    1. 回転補正（EXIF 対応漏れ向け）
    2. 推定解像度へのリサイズ（長辺 960px）
    3. CLAHE（砂地テクスチャ対応）
    戻り値はリサイズ後の BGR（推定専用、元解像度ではない）。
    """
    frame = bgr.copy()
    if rotate_code is not None:
        frame = cv2.rotate(frame, rotate_code)
    if (frame.shape[1], frame.shape[0]) != (meta.proc_width, meta.proc_height):
        frame = cv2.resize(frame, (meta.proc_width, meta.proc_height),
                           interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


# =============================================================================
# キーポイント抽出
# =============================================================================

def extract_keypoints(
    result,
    frame_idx: int,
    time_sec: float,
    fps: float,
    meta: VideoMeta,
) -> FrameResult:
    fr = FrameResult(frame=frame_idx, time_sec=time_sec, fps=fps)
    if not result.pose_landmarks:
        return fr

    lm_list = result.pose_landmarks[0]
    for name, idx in TARGET_KP.items():
        lm = lm_list[idx]
        valid = lm.visibility >= VISIBILITY_THRESHOLD
        fr.keypoints.append(KeypointRecord(
            frame=frame_idx,
            time_sec=time_sec,
            name=name,
            x_norm=float(lm.x),
            y_norm=float(lm.y),
            x_px=int(np.clip(lm.x, 0, 1) * meta.width),
            y_px=int(np.clip(lm.y, 0, 1) * meta.height),
            visibility=float(lm.visibility),
            valid=valid,
        ))
    return fr


# =============================================================================
# リリース瞬間検出
# =============================================================================

def detect_release_frames(
    frame_results: list[FrameResult],
    fps: float,
) -> list[int]:
    """
    両手首の速度ピークからリリース瞬間を検出。

    縦動画では Y 軸（上下方向）が長いため Y 成分が大きく出るが、
    正規化座標を使うため横動画と同一閾値で動作する。
    """
    times  = np.array([fr.time_sec for fr in frame_results])
    frames = np.array([fr.frame    for fr in frame_results])
    n      = len(times)

    wrist_speeds = np.zeros(n)
    for kp_name in ("right_wrist", "left_wrist"):
        xs = np.array([_get_kp(fr, kp_name, "x_norm") for fr in frame_results])
        ys = np.array([_get_kp(fr, kp_name, "y_norm") for fr in frame_results])
        vs = np.array([_get_vis(fr, kp_name)          for fr in frame_results])
        dx = np.gradient(xs, times, edge_order=1)
        dy = np.gradient(ys, times, edge_order=1)
        speed = np.sqrt(dx**2 + dy**2)
        wrist_speeds += speed * (vs >= VISIBILITY_THRESHOLD).astype(float)

    smooth_size = max(3, int(fps * 0.06))
    wrist_speeds = uniform_filter1d(wrist_speeds, size=smooth_size)

    min_dist = max(RELEASE_PEAK_MIN_DISTANCE, int(fps * 0.15))
    peaks, _ = find_peaks(
        wrist_speeds,
        height=RELEASE_PEAK_MIN_HEIGHT,
        distance=min_dist,
    )
    if len(peaks) == 0:
        return []

    best = peaks[np.argmax(wrist_speeds[peaks])]
    return [int(frames[best])]


# =============================================================================
# Kalman フィルタ + スプライン補間
# =============================================================================

class KalmanFilter1D:
    """定速モデルの 1D Kalman フィルタ（欠損補完・ノイズ低減）"""

    def __init__(self, q: float = 1e-3, r: float = 1e-2):
        self.Q = q
        self.R = r
        self.x = 0.0
        self.v = 0.0
        self.P = np.eye(2)

    def reset(self, x0: float, v0: float = 0.0):
        self.x, self.v = x0, v0
        self.P = np.eye(2)

    def predict(self, dt: float):
        self.x += self.v * dt
        self.P[0, 0] += dt * (self.P[1, 0] + self.P[0, 1]) + dt**2 * self.P[1, 1] + self.Q
        self.P[0, 1] += dt * self.P[1, 1]
        self.P[1, 0] += dt * self.P[1, 1]

    def update(self, z: float):
        S = self.P[0, 0] + self.R
        Kx = self.P[0, 0] / S
        Kv = self.P[1, 0] / S
        inn = z - self.x
        self.x += Kx * inn
        self.v += Kv * inn
        self.P[0, 0] -= Kx * self.P[0, 0]
        self.P[1, 0] -= Kv * self.P[0, 0]
        self.P[0, 1] -= Kx * self.P[0, 1]
        self.P[1, 1] -= Kv * self.P[0, 1]


def smooth_timeseries(
    frame_results: list[FrameResult],
    fps: float,
) -> list[FrameResult]:
    """
    各キーポイントに対して Kalman + CubicSpline でスムージング。
    frame_results をインプレースで更新して返す。
    """
    times = np.array([fr.time_sec for fr in frame_results])
    dt    = 1.0 / fps

    for kp_name in TARGET_KP:
        xs_raw = np.array([_get_kp(fr, kp_name, "x_norm") for fr in frame_results])
        ys_raw = np.array([_get_kp(fr, kp_name, "y_norm") for fr in frame_results])
        vs_raw = np.array([_get_vis(fr, kp_name)          for fr in frame_results])

        kf_x = KalmanFilter1D()
        kf_y = KalmanFilter1D()
        first_valid = np.where(vs_raw >= VISIBILITY_THRESHOLD)[0]
        if len(first_valid):
            fi = first_valid[0]
            kf_x.reset(xs_raw[fi])
            kf_y.reset(ys_raw[fi])

        xs_k = xs_raw.copy()
        ys_k = ys_raw.copy()
        for i in range(len(times)):
            kf_x.predict(dt)
            kf_y.predict(dt)
            if vs_raw[i] >= VISIBILITY_THRESHOLD:
                kf_x.update(xs_raw[i])
                kf_y.update(ys_raw[i])
            xs_k[i] = kf_x.x
            ys_k[i] = kf_y.x

        if len(times) >= 4:
            try:
                xs_s = np.clip(CubicSpline(times, xs_k)(times), 0.0, 1.0)
                ys_s = np.clip(CubicSpline(times, ys_k)(times), 0.0, 1.0)
            except Exception:
                xs_s, ys_s = xs_k, ys_k
        else:
            xs_s, ys_s = xs_k, ys_k

        for i, fr in enumerate(frame_results):
            for kp in fr.keypoints:
                if kp.name == kp_name:
                    kp.x_norm = float(xs_s[i])
                    kp.y_norm = float(ys_s[i])
                    break

    return frame_results


# =============================================================================
# 描画
# =============================================================================

SKELETON_PAIRS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("right_hip",      "right_knee"),
    ("left_shoulder",  "left_wrist"),
    ("right_shoulder", "right_wrist"),
]


def draw_skeleton(
    bgr: np.ndarray,
    fr: FrameResult,
    is_release: bool,
    in_window: bool,
    meta: VideoMeta,
) -> np.ndarray:
    """
    骨格・キーポイント・HUD を描画。
    縦動画（1080x1920）ではフォントサイズ・点サイズをスケーリング。
    """
    out = bgr.copy()
    h, w = out.shape[:2]
    kpm = {kp.name: kp for kp in fr.keypoints}

    # スケーリング係数（縦動画の 1920px 基準で拡大）
    scale      = max(w, h) / 1080.0
    font_scale = max(0.5, scale * 0.65)
    dot_r      = max(5, int(scale * 8))
    line_w     = max(2, int(scale * 2.5))

    # リリース窓内の薄いオーバーレイ
    if in_window and not is_release:
        ov = out.copy()
        cv2.rectangle(ov, (0, 0), (w, h), COLOR_WINDOW, -1)
        cv2.addWeighted(ov, 0.07, out, 0.93, 0, out)

    # リリース枠
    if is_release:
        bw = max(6, int(scale * 10))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), COLOR_RELEASE, bw)

    # スケルトンライン
    bone_c = COLOR_RELEASE if is_release else COLOR_BONE
    for a_n, b_n in SKELETON_PAIRS:
        if a_n not in kpm or b_n not in kpm:
            continue
        a, b = kpm[a_n], kpm[b_n]
        if a.valid and b.valid:
            cv2.line(out, (a.x_px, a.y_px), (b.x_px, b.y_px),
                     bone_c, line_w, cv2.LINE_AA)

    # キーポイント円
    for kp in fr.keypoints:
        if not kp.valid:
            continue
        c = COLOR_RELEASE if is_release else COLOR_KP
        cv2.circle(out, (kp.x_px, kp.y_px), dot_r,     c,       -1, cv2.LINE_AA)
        cv2.circle(out, (kp.x_px, kp.y_px), dot_r + 1, (0,0,0),  1, cv2.LINE_AA)

    # ラベル
    for kp in fr.keypoints:
        if not kp.valid:
            continue
        short = kp.name.replace("left_", "L.").replace("right_", "R.")
        cv2.putText(out, short,
                    (kp.x_px + dot_r + 3, kp.y_px - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)

    # HUD バー
    hud_h = max(44, int(scale * 48))
    cv2.rectangle(out, (0, 0), (w, hud_h), (0, 0, 0), -1)
    status_c = COLOR_RELEASE if is_release else (200, 200, 200)
    label    = (f"RELEASE  t={fr.time_sec:.3f}s" if is_release
                else f"t={fr.time_sec:.3f}s  frame={fr.frame}")
    cv2.putText(out, label, (12, int(hud_h * 0.72)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                status_c, 1, cv2.LINE_AA)

    return out


# =============================================================================
# CSV 出力
# =============================================================================

CSV_COLUMNS = [
    "frame", "time_sec", "name",
    "x_norm", "y_norm", "x_px", "y_px",
    "visibility", "valid", "release_flag", "in_window",
]


def export_csv(
    frame_results: list[FrameResult],
    path: Path,
    label: str = "",
):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for fr in frame_results:
            for kp in fr.keypoints:
                writer.writerow({
                    "frame":        kp.frame,
                    "time_sec":     f"{kp.time_sec:.4f}",
                    "name":         kp.name,
                    "x_norm":       f"{kp.x_norm:.5f}",
                    "y_norm":       f"{kp.y_norm:.5f}",
                    "x_px":         kp.x_px,
                    "y_px":         kp.y_px,
                    "visibility":   f"{kp.visibility:.4f}",
                    "valid":        int(kp.valid),
                    "release_flag": kp.release_flag,
                    "in_window":    kp.in_window,
                })
    tag = f"[{label}] " if label else ""
    print(f"  {tag}CSV 保存: {path}  ({len(frame_results)} フレーム)")


def export_release_window_csv(frame_results: list[FrameResult], path: Path):
    """リリース前後ウィンドウ内フレームのみ抽出した高密度 CSV。"""
    win = [fr for fr in frame_results
           if any(kp.in_window for kp in fr.keypoints)]
    if not win:
        print("  [Release Window] リリース未検出のためスキップ。")
        return
    export_csv(win, path, label="Release Window")
    print(f"  ウィンドウ内フレーム数: {len(win)}")


# =============================================================================
# ヘルパー
# =============================================================================

def _get_kp(fr: FrameResult, name: str, attr: str) -> float:
    for kp in fr.keypoints:
        if kp.name == name:
            return float(getattr(kp, attr))
    return 0.0


def _get_vis(fr: FrameResult, name: str) -> float:
    for kp in fr.keypoints:
        if kp.name == name:
            return kp.visibility
    return 0.0


def _rotate_code(angle: int) -> Optional[int]:
    return {90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(angle)


# =============================================================================
# メインパイプライン
# =============================================================================

def run_pipeline(
    input_video: Path,
    output_dir: Path,
    model_path: Path,
    device: str = "auto",
    output_video: bool = True,
    skip_preprocess: bool = False,
    rotate: int = 0,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_video.stem

    # -- 動画メタ取得 ----------------------------------------------------------
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        sys.exit(f"ERROR: 動画を開けません: {input_video}")
    meta = VideoMeta.from_cap(cap)
    cap.release()
    print(f"¥n動画情報: {meta}")

    rot_code = _rotate_code(rotate)
    if rot_code is not None:
        print(f"回転補正: {rotate}deg")

    actual_device = detect_device(device)
    print(f"実行デバイス: {actual_device.upper()}")

    # -- BlazePose 初期化 ------------------------------------------------------
    print("BlazePose モデルを読み込み中...")
    pose = build_pose_landmarker(model_path, device=actual_device, mode="VIDEO")
    print("  完了。")

    # -- Phase 1: 推定ループ ---------------------------------------------------
    cap = cv2.VideoCapture(str(input_video))
    frame_results: list[FrameResult] = []
    frame_idx = 0
    t0 = time.perf_counter()
    print("推定処理中...")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if time_ms == 0 and frame_idx > 0:
            time_ms = int(frame_idx * 1000.0 / meta.fps)
        time_sec = time_ms / 1000.0

        proc = (cv2.resize(bgr, (meta.proc_width, meta.proc_height),
                           interpolation=cv2.INTER_AREA)
                if skip_preprocess
                else preprocess_frame(bgr, meta, rot_code))

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
        result = pose.detect_for_video(mp_img, time_ms)

        fr = extract_keypoints(result, frame_idx, time_sec, meta.fps, meta)
        frame_results.append(fr)
        frame_idx += 1

        if frame_idx % 90 == 0:
            elapsed = time.perf_counter() - t0
            pct = frame_idx / max(meta.total_frames, 1) * 100
            eta = elapsed / frame_idx * max(0, meta.total_frames - frame_idx)
            print(f"  [{pct:4.0f}%] frame={frame_idx}/{meta.total_frames}"
                  f"  ETA={eta:.0f}s")

    cap.release()
    pose.close()
    print(f"推定完了: {len(frame_results)} フレーム  ({time.perf_counter()-t0:.1f}s)")

    if not frame_results:
        sys.exit("ERROR: フレームを 1 枚も取得できませんでした。")

    # -- Phase 2: リリース検出 -------------------------------------------------
    print("リリース瞬間を検出中...")
    release_frames = detect_release_frames(frame_results, meta.fps)
    if release_frames:
        fn2i = {fr.frame: i for i, fr in enumerate(frame_results)}
        for rf in release_frames:
            rt = frame_results[fn2i.get(rf, 0)].time_sec
            print(f"  検出: frame={rf}  t={rt:.3f}s")
    else:
        print("  未検出（--release-threshold を下げてみてください）。")

    win_size = int(meta.fps * RELEASE_WINDOW_SEC)
    rel_set  = set(release_frames)
    win_set: set[int] = set()
    for rf in release_frames:
        win_set.update(range(rf - win_size, rf + win_size + 1))

    for fr in frame_results:
        is_rel = fr.frame in rel_set
        in_win = fr.frame in win_set
        for kp in fr.keypoints:
            kp.release_flag = 1 if is_rel else 0
            kp.in_window    = 1 if in_win else 0

    # -- Phase 3: 補間・スムージング -------------------------------------------
    print("Kalman + スプライン補間中...")
    frame_results = smooth_timeseries(frame_results, meta.fps)
    print("  完了。")

    # -- Phase 4: CSV 出力 -----------------------------------------------------
    csv_all = output_dir / f"{stem}_keypoints.csv"
    csv_win = output_dir / f"{stem}_release_window.csv"
    print("CSV を書き出し中...")
    export_csv(frame_results, csv_all, label="All frames")
    export_release_window_csv(frame_results, csv_win)

    # -- Phase 5: オーバーレイ動画 ---------------------------------------------
    if output_video:
        vpath = output_dir / f"{stem}_skeleton.mp4"
        print(f"オーバーレイ動画を生成中 -> {vpath} ...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(vpath), fourcc, meta.fps, (meta.width, meta.height))

        cap2 = cv2.VideoCapture(str(input_video))
        for fr in frame_results:
            ret, bgr = cap2.read()
            if not ret:
                break
            if rot_code is not None:
                bgr = cv2.rotate(bgr, rot_code)
            for kp in fr.keypoints:
                kp.x_px = int(np.clip(kp.x_norm, 0, 1) * meta.width)
                kp.y_px = int(np.clip(kp.y_norm, 0, 1) * meta.height)
            drawn = draw_skeleton(bgr, fr, fr.frame in rel_set, fr.frame in win_set, meta)
            vw.write(drawn)
        cap2.release()
        vw.release()
        print(f"  動画保存完了: {vpath}")

    # -- 完了サマリー ----------------------------------------------------------
    print("¥n" + "=" * 52)
    print(f"  完了  出力先: {output_dir}")
    print(f"  総フレーム    : {len(frame_results)}")
    print(f"  リリース検出  : {release_frames}")
    print("=" * 52)


# =============================================================================
# モデルダウンロード
# =============================================================================

def download_model(dest: Path = MODEL_PATH):
    if dest.exists():
        print(f"モデルは既にあります: {dest}")
        return
    print(f"ダウンロード中: {MODEL_URL}")

    def _progress(block_num, block_size, total_size):
        done = block_num * block_size
        pct  = min(100, done * 100 // max(total_size, 1))
        bar  = "=" * (pct // 5) + "." * (20 - pct // 5)
        print(f"¥r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(MODEL_URL, dest, reporthook=_progress)
    print(f"¥n保存: {dest}")


# =============================================================================
# CLI
# =============================================================================

def main():
    global RELEASE_PEAK_MIN_HEIGHT
    parser = argparse.ArgumentParser(
        description="モルック投擲動画 骨格キーポイント抽出パイプライン v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python molkky_pose_pipeline.py --input throw.mp4
  python molkky_pose_pipeline.py --input throw.mp4 --no-video
  python molkky_pose_pipeline.py --input throw.mp4 --device gpu
  python molkky_pose_pipeline.py --input throw.mp4 --rotate 90
  python molkky_pose_pipeline.py --download-model
""",
    )
    parser.add_argument("--input", "-i", type=Path, help="入力動画 (MP4/MOV/AVI)")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./output"))
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"])
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="フレーム回転補正 [度]")
    parser.add_argument("--no-video", action="store_true",
                        help="オーバーレイ動画の出力をスキップ")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="CLAHE 前処理をスキップ")
    parser.add_argument("--release-threshold", type=float,
                        default=RELEASE_PEAK_MIN_HEIGHT,
                        help=f"リリース検出速度閾値 (default: {RELEASE_PEAK_MIN_HEIGHT})")
    parser.add_argument("--download-model", action="store_true",
                        help="BlazePose Heavy モデルをダウンロードして終了")
    args = parser.parse_args()

    if args.download_model:
        download_model(args.model)
        return

    RELEASE_PEAK_MIN_HEIGHT = args.release_threshold

    if not args.input:
        parser.error("--input を指定してください。")
    if not args.input.exists():
        sys.exit(f"ERROR: ファイルが見つかりません: {args.input}")
    if not args.model.exists():
        print(f"モデルが見つかりません: {args.model}")
        print(f"  python {Path(__file__).name} --download-model")
        sys.exit(1)

    run_pipeline(
        input_video=args.input,
        output_dir=args.output_dir,
        model_path=args.model,
        device=args.device,
        output_video=not args.no_video,
        skip_preprocess=args.no_preprocess,
        rotate=args.rotate,
    )


if __name__ == "__main__":
    main()
