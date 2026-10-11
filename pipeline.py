"""Computer-vision pipeline: detect + track players/ball, split teams, estimate formation,
then ask a vision LLM (Claude) for coaching feedback. Falls back to heuristics with no API key."""
import base64
import json
import math
import os
import re

import cv2
import numpy as np
from sklearn.cluster import KMeans

import boxtracker
import camera
import fieldmask
import sports
import vocab

MODEL = os.environ.get("ANALYSIS_MODEL", "claude-sonnet-5-5")
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
VERSION = "10"       # bump whenever results change shape or meaning, so stale cached results are not reused
TARGET_FPS = 6          # quick-scan analysis fps (tracking runs at this rate)
MAX_SECONDS = int(os.environ.get("MAX_SECONDS", 120))   # cap clip length analysed (seconds)
KEYFRAMES = 6           # frames sent to the vision model
# Deep scan: slower, more accurate. More frames per second, higher resolution, a dedicated ball pass.
DEEP_FPS = int(os.environ.get("DEEP_FPS", 12))
DEEP_MAX_SECONDS = int(os.environ.get("DEEP_MAX_SECONDS", 60))
PERSON_MIN_CONF = 0.20  # people below this are noise
BALL_MIN_CONF = 0.06    # the ball is small and blurry, so accept weak detections and judge them over time


# ----------------------------------------------------------------- helpers
def _torso_color(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 8 or w < 4:
        return None
    crop = frame[y1 + int(h * 0.15): y1 + int(h * 0.55), x1 + int(w * 0.2): x2 - int(w * 0.2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # drop grass-green pixels: green-ish hue, saturated and fairly dark. Bright shirts
    # (even lime/yellow ones with a similar hue) have a high value, so they survive.
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 90) & (hsv[..., 2] < 175)
    px = crop[~grass]
    if len(px) < 10:
        px = crop.reshape(-1, 3)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    return lab.mean(axis=0)


def _cone_like(frame, box):
    """True when the middle of a 'ball' detection is vividly coloured. Training cones, markers and discs are
    saturated yellow, orange or lime; a football's centre is white, grey or dark. Real balls in a vivid colour
    are rare enough in match footage that this is a good trade, and the score is only reduced, not zeroed."""
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 3 or h < 3:
        return False
    cx1, cy1 = x1 + int(w * 0.25), y1 + int(h * 0.25)
    cx2, cy2 = max(cx1 + 1, x2 - int(w * 0.25)), max(cy1 + 1, y2 - int(h * 0.25))
    crop = frame[max(0, cy1):cy2, max(0, cx1):cx2]
    if crop.size == 0:
        return False
    return float(np.median(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 1])) > 105


def _shorts_color(frame, box):
    """Lower-body colour (shorts). A second kit cue: two teams in similar shirts often differ here."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = y2 - y1, x2 - x1
    if h < 16 or w < 6:
        return None
    crop = frame[y1 + int(h * 0.58): y1 + int(h * 0.82), x1 + int(w * 0.2): x2 - int(w * 0.2)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 85) & (hsv[..., 1] > 90) & (hsv[..., 2] < 175)
    px = crop[~grass]
    if len(px) < 8:
        px = crop.reshape(-1, 3)
    return cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)


def _person_crop(frame, box, size=(64, 128)):
    """A small normalised picture of one player, kept for appearance comparison."""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if y2 - y1 < 24 or x2 - x1 < 10:
        return None
    return cv2.resize(frame[y1:y2, x1:x2], size)


def _formation(points_xy, attack_axis="x"):
    """Estimate a formation string (e.g. 4-4-2) from average outfield positions."""
    pts = np.array(points_xy)
    n = len(pts)
    if n < 5:
        return "unclear"
    depth = pts[:, 0]
    k = 4 if n >= 9 else 3 if n >= 6 else 2
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(depth.reshape(-1, 1))
    order = list(np.argsort(km.cluster_centers_.ravel()))
    counts = [int(np.sum(km.labels_ == i)) for i in order]
    # a lone player at one end of the depth axis is the goalkeeper: drop it and
    # order the lines from the goalkeeper's end outward (defence -> attack)
    if n >= 9 and counts[0] == 1 and counts[-1] != 1:
        counts = counts[1:]
    elif n >= 9 and counts[-1] == 1 and counts[0] != 1:
        counts = counts[::-1][1:]
    elif n >= 9 and counts[0] == 1 and counts[-1] == 1:
        # both ends singleton: keeper vs. lone striker - keeper is the end further from the next line
        gap_lo = km.cluster_centers_.ravel()[order[1]] - km.cluster_centers_.ravel()[order[0]]
        gap_hi = km.cluster_centers_.ravel()[order[-1]] - km.cluster_centers_.ravel()[order[-2]]
        counts = counts[1:] if gap_lo >= gap_hi else counts[::-1][1:]
    return "-".join(str(c) for c in counts)


def _lines(points_xy, k, axis="x"):
    """Group players into k depth bands along one axis and return the counts, low to high
    (for example '3-2' for a basketball half). Used by sports without a goalkeeper rule."""
    pts = np.array(points_xy)
    n = len(pts)
    if n < 2:
        return "unclear"
    k = max(1, min(k, n))  # never more bands than players
    if k == 1:
        return str(n)
    depth = pts[:, 0 if axis == "x" else 1]
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(depth.reshape(-1, 1))
    order = np.argsort(km.cluster_centers_.ravel())
    return "-".join(str(int(np.sum(km.labels_ == i))) for i in order)


def _ring(points_xy, words=("close", "deep")):
    """Cricket/baseball-style shape: split players into an inner ring and an outer ring by their
    distance from the group's centre, and return e.g. 'N close / M deep' (or infield / outfield)."""
    pts = np.array(points_xy)
    if len(pts) < 3:
        return "unclear"
    centre = np.median(pts, axis=0)
    radius = np.hypot(*(pts - centre).T)
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(radius.reshape(-1, 1))
    inner = int(np.argmin(km.cluster_centers_.ravel()))
    n_in = int(np.sum(km.labels_ == inner))
    return f"{n_in} {words[0]} / {len(pts) - n_in} {words[1]}"


def _tennis(points_xy, axis="y"):
    """One side of a tennis court: singles, or how the doubles pair are arranged."""
    pts = np.array(points_xy)
    if len(pts) <= 1:
        return "Singles"
    depth = pts[:, 0 if axis == "x" else 1]
    return "One up, one back" if depth.max() - depth.min() > 0.12 else "Level"


def estimate_formation(points_xy, profile):
    """Describe a team's shape in the terms of its sport (soccer lines, basketball bands, ...)."""
    method, axis = profile["formation"], profile["depth_axis"]
    if method == "soccer":
        return _formation(points_xy)
    if method == "ring":
        return _ring(points_xy, profile.get("ring_words", ("close", "deep")))
    if method == "tennis":
        return _tennis(points_xy, axis)
    return _lines(points_xy, profile["lines"], axis)


def _b64_jpeg(img, width=1024):
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode()


def _frame_step(fps, deep):
    """Analyse every Nth video frame. Deep scan rounds the step UP so it never exceeds its fps budget."""
    if deep:
        return max(1, math.ceil(fps / DEEP_FPS - 1e-6))
    return max(1, int(round(fps / TARGET_FPS)))


# ----------------------------------------------------------------- motion detector (top-down / aerial)
# A person seen from straight above is a small blob, which a model trained on side-on photos misses.
# For a steady camera there is a better signal: whatever moves against the empty-scene background.
MOTION_WIDTH = 960          # work at this width; boxes are stored normalised, so scale does not matter
MOTION_THRESHOLD = 30       # how different from the background a pixel must be to count as "moving"


def _read_frame_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    return frame if ok else None


def _small(frame):
    h, w = frame.shape[:2]
    if w > MOTION_WIDTH:
        frame = cv2.resize(frame, (MOTION_WIDTH, int(h * MOTION_WIDTH / w)))
    return cv2.GaussianBlur(frame, (5, 5), 0)


ALIGN_WIDTH = 320           # camera-motion estimation runs on a 320 px wide grey copy


def _gray_small(img):
    h, w = img.shape[:2]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (ALIGN_WIDTH, max(8, int(h * ALIGN_WIDTH / w)))).astype(np.float32)


def _translate(img, dx, dy):
    """Move the picture's content by (-dx, -dy): undoes a camera drift of (dx, dy)."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), borderMode=cv2.BORDER_REPLICATE)


def _motion_background(path, n=30):
    """The empty scene: median of n frames spread across the clip, after lining the frames up with each other
    (a drone never holds perfectly still). Moving people drop out of the median. Returns (background, grey copy)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    last = min(total, int(MAX_SECONDS * (cap.get(cv2.CAP_PROP_FPS) or 25))) - 1
    shots = [_small(f) for f in (_read_frame_at(cap, int(i)) for i in np.linspace(0, max(0, last), n))
             if f is not None]
    cap.release()
    if len(shots) < 5:
        return None, None
    ref = shots[len(shots) // 2]
    ref_g = _gray_small(ref)
    win = cv2.createHanningWindow((ref_g.shape[1], ref_g.shape[0]), cv2.CV_32F)
    k = ref.shape[1] / ALIGN_WIDTH
    aligned = []
    for s in shots:
        (dx, dy), strength = cv2.phaseCorrelate(ref_g, _gray_small(s), win)
        if strength >= 0.03 and abs(dx * k) < 0.35 * ref.shape[1] and abs(dy * k) < 0.35 * ref.shape[0]:
            aligned.append(_translate(s, dx * k, dy * k))
    if len(aligned) < 4:
        aligned = shots
    bg = np.median(np.stack(aligned), axis=0).astype(np.uint8)
    return bg, _gray_small(bg)


def _blobs_aligned(small, bg, bg_gray, win):
    """Line this frame up with the background, then find moving blobs. Returns (blobs, (dx, dy)) where the blobs
    are in the background's coordinates and (dx, dy) is the camera drift in small-frame pixels, or (None, None)
    when the frame cannot be lined up (camera zoomed or swung too far to compare)."""
    k = small.shape[1] / ALIGN_WIDTH
    (dx, dy), strength = cv2.phaseCorrelate(bg_gray, _gray_small(small), win)
    dx, dy = dx * k, dy * k
    h, w = small.shape[:2]
    if strength < 0.03 or abs(dx) > 0.35 * w or abs(dy) > 0.35 * h:
        return None, None
    valid = (max(0.0, -dx) + 4, max(0.0, -dy) + 4, min(w, w - dx) - 4, min(h, h - dy) - 4)   # real picture area
    return _motion_blobs(_translate(small, dx, dy), bg, valid), (dx, dy)


def _motion_blobs(small, bg, valid=None):
    """Find moving blobs of roughly person size. Returns [(x1, y1, x2, y2, lab_colour)] in small-frame
    pixels, or None when most of the frame changed (the camera moved, so subtraction is meaningless)."""
    diff = cv2.absdiff(small, bg).max(axis=2)
    changed = diff > MOTION_THRESHOLD                    # pixels that really differ from the background
    mask = changed.astype(np.uint8) * 255
    if mask.mean() / 255.0 > 0.12:
        return None
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    lab_frame = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = small.shape[:2]
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        # person-sized only: a player seen from above is tens of pixels, a bare patch of turf is hundreds
        if area < 40 or area > 0.0035 * w * h or not (0.15 < bw / max(1, bh) < 6.5):
            continue
        if valid is not None and (x < valid[0] or y < valid[1] or x + bw > valid[2] or y + bh > valid[3]):
            continue                                   # touches the strip the camera drift exposed
        sl = (slice(y, y + bh), slice(x, x + bw))
        # Kit colour = pixels that changed AND whose hue differs from the background there.
        # The padded blob also holds grass and the player's shadow, which keep the background's hue.
        own = changed[sl] & (labels[sl] == i)
        f, b = lab_frame[sl][own], lab_bg[sl][own]
        if len(f) == 0:
            continue
        hue_gap = np.hypot(f[:, 1] - b[:, 1], f[:, 2] - b[:, 2])
        kit = f[hue_gap > 12]
        lab = (kit if len(kit) >= 3 else f).mean(axis=0)
        out.append((x, y, x + bw, y + bh, lab))
    return out


class _CentroidTracker:
    """Very small multi-object tracker: match each blob to the nearest track seen recently."""

    def __init__(self, max_dist, max_missing=6):
        self.max_dist, self.max_missing = max_dist, max_missing
        self.tracks, self.next_id = {}, 1        # id -> [cx, cy, frames_since_seen]

    def update(self, dets):
        cands = [(np.hypot(c[0] - d[0], c[1] - d[1]), tid, j)
                 for tid, c in self.tracks.items() for j, d in enumerate(dets)]
        used_t, used_d, ids = set(), set(), [None] * len(dets)
        for dist, tid, j in sorted(cands):
            if dist > self.max_dist or tid in used_t or j in used_d:
                continue
            used_t.add(tid); used_d.add(j); ids[j] = tid
        for tid in list(self.tracks):
            if tid in used_t:
                continue
            self.tracks[tid][2] += 1
            if self.tracks[tid][2] > self.max_missing:
                del self.tracks[tid]
        for j, d in enumerate(dets):
            if ids[j] is None:
                ids[j] = self.next_id; self.next_id += 1
            self.tracks[ids[j]] = [d[0], d[1], 0]
        return ids


def _track_motion(path, progress, deep=False):
    bg, bg_gray = _motion_background(path)
    cap = cv2.VideoCapture(path)
    if bg is None or not cap.isOpened():
        raise RuntimeError("That file could not be read as a video. Try an MP4 or MOV clip.")
    win = cv2.createHanningWindow((bg_gray.shape[1], bg_gray.shape[0]), cv2.CV_32F)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sh, sw = bg.shape[:2]
    step = _frame_step(fps, deep)
    max_frames = int((min(MAX_SECONDS, DEEP_MAX_SECONDS) if deep else MAX_SECONDS) * fps)
    key_every = max(1, min(total, max_frames) // (KEYFRAMES * step)) * step
    trk = _CentroidTracker(max_dist=0.045 * sw * (6.0 / (DEEP_FPS if deep else TARGET_FPS)) ** 0.5 + 6)
    frames, keyframe_imgs, tracks = [], [], {}
    idx = 0
    progress(5, "Tracking movers (aerial mode)")
    while True:
        ok, frame = cap.read()
        if not ok or idx > max_frames:
            break
        if idx % step == 0:
            t = idx / fps
            rec = {"t": round(t, 2), "p": [], "ball": None}
            blobs, off = _blobs_aligned(_small(frame), bg, bg_gray, win)
            if blobs:
                dx, dy = off
                rec["cam"] = [round(dx / sw, 4), round(dy / sh, 4)]
                # tracking runs in the background's (steady) coordinates; boxes are stored in the frame's own
                centres = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in blobs]
                for tid, b, c in zip(trk.update(centres), blobs, centres):
                    fx1, fy1, fx2, fy2 = b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy
                    tr = tracks.setdefault(tid, {"colors": [], "colors2": [], "crops": [], "pts": []})
                    tr["colors"].append(b[4])
                    tr["pts"].append((t, (fx1 + fx2) / 2 / sw, fy2 / sh))
                    rec["p"].append([tid, round(fx1 / sw, 4), round(fy1 / sh, 4),
                                     round(fx2 / sw, 4), round(fy2 / sh, 4)])
            frames.append(rec)
            if (idx // step) % max(1, key_every // step) == 0 and len(keyframe_imgs) < KEYFRAMES:
                keyframe_imgs.append((idx / fps, frame.copy(), rec))
            progress(5 + 55 * min(1.0, idx / max(1, min(total, max_frames))), "Tracking movers (aerial mode)")
        idx += 1
    cap.release()
    return {"fps": fps, "W": W, "H": H, "frames": frames, "tracks": tracks, "keys": keyframe_imgs,
            "step_fps": fps / step}


# ----------------------------------------------------------------- choosing a detector
def _sample_frames_for_probe(path, n=8):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = [f for f in (_read_frame_at(cap, int((k + 0.5) * total / n)) for k in range(n)) if f is not None]
    cap.release()
    return frames


def choose_detector(path, profile, progress=lambda p, m: None):
    """Try the fast detector first, then a bigger one at higher resolution, then the motion detector,
    and keep the first that finds enough people (or the best one if none do)."""
    from ultralytics import YOLO

    target = max(2, min(8, int(0.4 * profile["max_players"])))
    frames = _sample_frames_for_probe(path)
    if not frames:
        raise RuntimeError("That file could not be read as a video. Try an MP4 or MOV clip.")
    progress(3, "Choosing the best detector for this clip")
    tried = []
    for weights, imgsz in ((YOLO_WEIGHTS, 640), ("yolov8m.pt", 1280)):
        model = YOLO(weights)
        counts = [len(model.predict(f, imgsz=imgsz, conf=0.2, classes=[0], verbose=False)[0].boxes) for f in frames]
        median = float(np.median(counts))
        cfg = {"kind": "yolo", "weights": weights, "imgsz": imgsz, "name": f"{weights[:-3]}@{imgsz}"}
        tried.append((median, cfg))
        if median >= target:
            return cfg
    bg, bg_gray = _motion_background(path, n=20)
    if bg is not None:
        win = cv2.createHanningWindow((bg_gray.shape[1], bg_gray.shape[0]), cv2.CV_32F)
        counts = []
        for f in frames:
            blobs, _ = _blobs_aligned(_small(f), bg, bg_gray, win)
            counts.append(len(blobs) if blobs is not None else 0)
        motion = float(np.median(counts))
        best_yolo = max(tried, key=lambda t: t[0])
        if motion >= 3 and motion > best_yolo[0]:
            return {"kind": "motion", "name": "motion (aerial)"}
    return max(tried, key=lambda t: t[0])[1]


# ----------------------------------------------------------------- tracking
def track_video(path, progress=lambda p, m: None, detector=None, deep=False):
    from ultralytics import YOLO

    detector = detector or {"kind": "yolo", "weights": YOLO_WEIGHTS, "imgsz": 640}
    if detector["kind"] == "motion":
        return _track_motion(path, progress, deep)
    model = YOLO(detector["weights"])
    # Deep scan runs a second, detection-only pass for the ball. The ball is small and often low-confidence,
    # and the tracker throws away weak detections, so it needs its own model instance and a low threshold.
    ball_model = YOLO(detector["weights"]) if deep else None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("That file could not be read as a video. Try an MP4 or MOV clip.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = _frame_step(fps, deep)
    max_frames = int((min(MAX_SECONDS, DEEP_MAX_SECONDS) if deep else MAX_SECONDS) * fps)
    imgsz = detector["imgsz"]
    stage = "Deep scan: tracking every player and the ball" if deep else "Tracking players"

    frames, keyframe_imgs, thumbs = [], [], []
    tracks = {}     # id -> dict(colors=[], colors2=[], crops=[], pts=[])
    thumb_every = fieldmask.keep_every(max(1, min(total, max_frames) // step))
    person_conf = PERSON_MIN_CONF if deep else (0.25 if imgsz <= 640 else 0.2)
    tracker = boxtracker.BoxTracker(fps / step, high_conf=person_conf, low_conf=max(0.08, person_conf - 0.1),
                                    size=(W, H))
    camtrack = camera.CameraTracker(W, H)   # pan AND zoom of the camera: a phone on the sideline never holds still
    idx = 0
    key_every = max(1, min(total, max_frames) // (KEYFRAMES * step)) * step
    progress(5, stage)
    while True:
        ok, frame = cap.read()
        if not ok or idx > max_frames:
            break
        if idx % step == 0:
            M = camtrack.update(frame)
            cam = camtrack.centre_shift()
            # One plain detection pass, then our own association (see boxtracker.py: the stock tracker drops
            # small, bunched-up players because it needs two frames of box overlap before it reports anyone).
            res = model.predict(frame, classes=[0] if deep else [0, 32], conf=tracker.low,
                                imgsz=imgsz, verbose=False)[0]
            t = idx / fps
            rec = {"t": round(t, 2), "p": [], "ball": None, "cam": [round(cam[0], 4), round(cam[1], 4)],
                   "M": [round(float(v), 5) for v in M[:2].ravel()]}
            ball_cands = []                               # (confidence, cx, cy, width/height) in normalised coords
            people, pcols = [], []
            if res.boxes is not None and len(res.boxes):
                for box, cid, bconf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box
                    if int(cid) == 32:
                        if float(bconf) >= person_conf:
                            ball_cands.append((float(bconf), float((x1 + x2) / 2 / W), float((y1 + y2) / 2 / H),
                                               float((x2 - x1) / max(1.0, y2 - y1)),
                                               1.0 if _cone_like(frame, box) else 0.0))
                        continue
                    if (x2 - x1) < 3 or (y2 - y1) < 6:    # slivers are not people
                        continue
                    people.append((float(x1), float(y1), float(x2), float(y2), float(bconf)))
                    pcols.append(_torso_color(frame, box))
            ids = tracker.update(people, pcols, M)
            for (x1, y1, x2, y2, _c), tid, col in zip(people, ids, pcols):
                if tid is None:
                    continue
                tr = tracks.setdefault(tid, {"colors": [], "colors2": [], "crops": [], "pts": []})
                if col is not None:
                    tr["colors"].append(col)
                if deep:
                    box = (x1, y1, x2, y2)
                    col2 = _shorts_color(frame, box)
                    if col2 is not None:
                        tr["colors2"].append(col2)
                    if len(tr["crops"]) < 4 and len(tr["pts"]) % 14 == 1:
                        crop = _person_crop(frame, box)
                        if crop is not None:
                            tr["crops"].append(crop)
                fx, fy = float((x1 + x2) / 2 / W), float(y2 / H)
                tr["pts"].append((t, fx, fy))
                rec["p"].append([tid, round(x1 / W, 4), round(y1 / H, 4), round(x2 / W, 4), round(y2 / H, 4)])
            if deep:
                bres = ball_model.predict(frame, classes=[32], conf=BALL_MIN_CONF, imgsz=imgsz, verbose=False)[0]
                for b, c in zip(bres.boxes.xyxy.cpu().numpy(), bres.boxes.conf.cpu().numpy()):
                    ball_cands.append((float(c), float((b[0] + b[2]) / 2 / W), float((b[1] + b[3]) / 2 / H),
                                       float((b[2] - b[0]) / max(1.0, b[3] - b[1])),
                                       1.0 if _cone_like(frame, b) else 0.0))
            # keep several candidates per frame; finalize_ball() later picks the one plausible trajectory
            rec["ball_cands"] = [[round(c, 3), round(x, 4), round(y, 4), round(a, 2), int(k)]
                                 for c, x, y, a, k in sorted(ball_cands, reverse=True)[:6] if c >= 0.10]
            frames.append(rec)
            thumbs.append(fieldmask.thumb(frame) if (len(frames) - 1) % thumb_every == 0 else None)
            if (idx // step) % max(1, key_every // step) == 0 and len(keyframe_imgs) < KEYFRAMES:
                keyframe_imgs.append((idx / fps, frame.copy(), rec))
            progress(5 + 55 * min(1.0, idx / max(1, min(total, max_frames))), stage)
        idx += 1
    cap.release()
    # a "track" seen in fewer than 3 analysed frames is a flicker, not a person
    flicker = {tid for tid, tr in tracks.items() if len(tr["pts"]) < 3}
    if flicker:
        for tid in flicker:
            tracks.pop(tid, None)
        for rec in frames:
            rec["p"] = [p for p in rec["p"] if p[0] not in flicker]
    return {"fps": fps, "W": W, "H": H, "frames": frames, "tracks": tracks, "keys": keyframe_imgs,
            "step_fps": fps / step, "thumbs": thumbs}


def _drop_static_candidates(frames, min_count=8, min_span=0.5):
    """Remove 'ball' detections that sit in one place across much of the clip. A training cone, a penalty
    spot or a white mark gets detected as a ball again and again; a real ball moves, or at least is not
    parked in exactly the same spot from the start of the clip to the end."""
    n = len(frames)

    def cell(f, c):
        # position on the pitch, not on the screen: subtract how far the camera has drifted since the start
        cam = f.get("cam", (0.0, 0.0))
        return round((c[1] - cam[0]) / 0.025), round((c[2] - cam[1]) / 0.025)

    cells = {}
    for i, f in enumerate(frames):
        for c in f.get("ball_cands", []):
            cells.setdefault(cell(f, c), set()).add(i)
    static = set()
    for (cx, cy), idxs in cells.items():
        around = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                around |= cells.get((cx + dx, cy + dy), set())
        if len(around) >= min_count and (max(around) - min(around)) >= min_span * n:
            static.add((cx, cy))
    dropped = 0
    for f in frames:
        keep = [c for c in f.get("ball_cands", []) if cell(f, c) not in static]
        dropped += len(f.get("ball_cands", [])) - len(keep)
        f["ball_cands"] = keep
    return dropped


def _still_places(path, step_fps, win_s=1.2, eps=0.012):
    """Spots (in camera-stabilised coordinates) where the chosen 'ball' sat essentially motionless for
    win_s seconds. Real balls are not held perfectly still while the play goes on around them."""
    L = max(3, int(round(win_s * step_fps)))
    places, i = [], 0
    while i + L <= len(path):
        seg = [s for s in path[i:i + L] if s is not None]
        if len(seg) >= 0.8 * L:
            xs, ys = [s[4] for s in seg], [s[5] for s in seg]
            if max(xs) - min(xs) < eps and max(ys) - min(ys) < eps:
                places.append((float(np.mean(xs)), float(np.mean(ys))))
                i += L
                continue
        i += 1
    return places


def _ban_places(frames, places, radius=0.02):
    """Remove every candidate near a banned spot, in every frame."""
    removed = 0
    for f in frames:
        cam = f.get("cam", (0.0, 0.0))
        keep = []
        for c in f.get("ball_cands", []):
            sx, sy = c[1] - cam[0], c[2] - cam[1]
            if any(math.hypot(sx - px, sy - py) < radius for px, py in places):
                removed += 1
            else:
                keep.append(c)
        f["ball_cands"] = keep
    return removed


def _best_ball_path(frames, step_fps):
    """Viterbi: choose one candidate (or 'not visible') per frame so that the ball follows a plausible path.
    Strong detections score well, big jumps between frames are penalised, and losing the ball is allowed."""
    n = len(frames)
    dt = 1.0 / max(step_fps, 1.0)
    reach = 0.10 * dt / 0.1 + 0.03                      # how far a ball can plausibly travel between frames
    MISS = None

    def stab(f, c):
        """(conf, x, y, aspect, stabilised x, stabilised y) so jumps are judged on the pitch, not the screen."""
        cam = f.get("cam", (0.0, 0.0))
        asp = c[3] if len(c) > 3 else 1.0
        vivid = c[4] if len(c) > 4 else 0
        # fold the "looks like a cone" evidence into the score here; indexes 4/5 stay the stabilised position
        conf = c[0] * (0.15 if vivid else 1.0) * (0.5 if (asp > 1.8 or asp < 0.5) else 1.0)
        return (conf, c[1], c[2], asp, c[1] - cam[0], c[2] - cam[1])

    states = [[MISS] + [stab(f, c) for c in f.get("ball_cands", [])] for f in frames]
    score = [[0.0] * len(s) for s in states]
    back = [[0] * len(s) for s in states]

    def emit(c):
        return 0.12 if c is MISS else c[0]     # c[0] already carries the colour and shape penalties

    def trans(a, b):
        if a is MISS and b is MISS:
            return 0.0
        if a is MISS or b is MISS:
            return -0.25 if a is MISS else -0.05
        d = math.hypot(a[4] - b[4], a[5] - b[5])
        return -2.0 * d if d <= reach else -2.5

    for j, c in enumerate(states[0]):
        score[0][j] = emit(c)
    for i in range(1, n):
        for j, c in enumerate(states[i]):
            best, arg = -1e9, 0
            for k, p in enumerate(states[i - 1]):
                v = score[i - 1][k] + trans(p, c)
                if v > best:
                    best, arg = v, k
            score[i][j], back[i][j] = best + emit(c), arg
    j = int(np.argmax(score[-1]))
    path = [None] * n
    for i in range(n - 1, -1, -1):
        path[i] = states[i][j]
        j = back[i][j]
    return path


def finalize_ball(data):
    """Clean the ball track and say honestly how often it was seen.

    Steps: (1) drop detections parked in one spot all clip (cones, markings), (2) pick the most plausible
    trajectory through the remaining candidates, allowing the ball to be out of sight, (3) fill short gaps.
    The ball is small and often hidden, so coverage well below 100% is normal and expected."""
    frames = data["frames"]
    n = len(frames)
    if n == 0:
        return {"seen": 0.0, "filled": 0.0, "static_dropped": 0}
    dropped = 0
    if any("ball_cands" in f for f in frames):
        dropped = _drop_static_candidates(frames)
        fps_ = data.get("step_fps", 6)
        path = _best_ball_path(frames, fps_)
        for _ in range(5):
            # a ball moves relative to the ground, a cone does not: anything held still for over a
            # second is banned at that spot in every frame, then the path is chosen again without it
            places = _still_places(path, fps_)
            if not places:
                break
            dropped += _ban_places(frames, places)
            path = _best_ball_path(frames, fps_)
        for f, c in zip(frames, path):
            f["ball"] = [c[1], c[2]] if c is not None else None
            if c is not None:
                f["ball_conf"] = c[0]
            f.pop("ball_cands", None)       # (the camera-drift value stays until the report has used it)
    xs = np.array([f["ball"][0] if f.get("ball") else np.nan for f in frames])
    ys = np.array([f["ball"][1] if f.get("ball") else np.nan for f in frames])
    # 1) outliers: far from the local median position
    for i in range(n):
        if np.isnan(xs[i]):
            continue
        lo, hi = max(0, i - 3), min(n, i + 4)
        wx, wy = xs[lo:hi], ys[lo:hi]
        if np.sum(~np.isnan(wx)) >= 3 and (abs(xs[i] - np.nanmedian(wx)) > 0.2 or abs(ys[i] - np.nanmedian(wy)) > 0.2):
            xs[i] = ys[i] = np.nan
            frames[i]["ball"] = None
    seen = float(np.mean(~np.isnan(xs)))
    # 2) fill gaps of up to ~0.6 s between two real detections
    max_gap = max(2, int(round(0.6 * data.get("step_fps", 6))))
    known = np.where(~np.isnan(xs))[0]
    for a, b in zip(known[:-1], known[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for k in range(1, gap + 1):
                r = k / (gap + 1)
                frames[a + k]["ball"] = [round(float(xs[a] + (xs[b] - xs[a]) * r), 4),
                                         round(float(ys[a] + (ys[b] - ys[a]) * r), 4)]
                frames[a + k]["ball_est"] = True
    filled = float(np.mean([bool(f.get("ball")) for f in frames]))
    return {"seen": round(seen, 3), "filled": round(filled, 3), "static_dropped": dropped}


def stitch_tracks(data, max_gap_s=1.6):
    """Merge fragments of one player. When the tracker loses someone behind another player it issues a new ID
    on return, so one person can become three. A track that starts shortly after another ends, close to where
    it ended and with a similar kit colour, is the same person."""
    tracks = data["tracks"]
    info = {i: {"s": t["pts"][0][0], "e": t["pts"][-1][0], "p0": t["pts"][0][1:], "p1": t["pts"][-1][1:],
                "c": np.mean(t["colors"], axis=0) if t["colors"] else None}
            for i, t in tracks.items() if t["pts"]}
    order = sorted(info, key=lambda i: info[i]["s"])
    parent = {i: i for i in info}
    taken = set()                                   # tracks that already have a successor

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for b in order:
        best, best_cost = None, 1e9
        for a in order:
            if a == b or a in taken or info[a]["e"] >= info[b]["s"]:
                continue
            gap = info[b]["s"] - info[a]["e"]
            if gap > max_gap_s:
                continue
            dist = math.hypot(info[a]["p1"][0] - info[b]["p0"][0], info[a]["p1"][1] - info[b]["p0"][1])
            if dist > 0.09 + 0.05 * gap:
                continue
            ca, cb = info[a]["c"], info[b]["c"]
            cdist = float(np.linalg.norm(ca - cb)) if ca is not None and cb is not None else 0.0
            if cdist > 28:
                continue
            cost = dist * 10 + cdist / 10 + gap
            if cost < best_cost:
                best, best_cost = a, cost
        if best is not None and root(best) != root(b):
            parent[b] = root(best)
            taken.add(best)
    remap = {i: root(i) for i in info}
    merged = sum(1 for i in info if remap[i] != i)
    if not merged:
        return 0
    new_tracks = {}
    for i in order:
        r = remap[i]
        tr = tracks[i]
        dst = new_tracks.setdefault(r, {"colors": [], "colors2": [], "crops": [], "pts": []})
        dst["colors"] += tr["colors"]
        dst["colors2"] += tr.get("colors2", [])
        dst["crops"] += tr.get("crops", [])
        dst["pts"] += tr["pts"]
    for tr in new_tracks.values():
        tr["pts"].sort(key=lambda p: p[0])
        tr["crops"] = tr["crops"][:6]
    data["tracks"] = new_tracks
    for rec in data["frames"]:
        seen, out = set(), []
        for p in rec["p"]:
            nid = remap.get(p[0], p[0])
            if nid in seen:                          # never two boxes with one ID in a frame
                continue
            seen.add(nid)
            out.append([nid, *p[1:]])
        rec["p"] = out
    return merged


# ----------------------------------------------------------------- teams / stats
def _lab_hex(lab):
    """Convert one OpenCV LAB colour to a '#rrggbb' string."""
    bgr = cv2.cvtColor(np.array(lab).astype(np.uint8).reshape(1, 1, 3), cv2.COLOR_LAB2BGR)[0, 0]
    return "#%02x%02x%02x" % (int(bgr[2]), int(bgr[1]), int(bgr[0]))


def _embed_tracks(tracks, ids):
    """Appearance embeddings from CLIP's image encoder (already loaded for sport detection). Two players in
    similar-looking kit can still differ in pattern, bibs or layering, which a single average colour misses."""
    import torch
    from PIL import Image

    import sportdetect
    model, proc = sportdetect._load()
    vecs = {}
    for i in ids:
        crops = tracks[i].get("crops") or []
        if not crops:
            continue
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with torch.no_grad():
            out = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
            f = out if hasattr(out, "shape") else out.pooler_output
            f = torch.nn.functional.normalize(f, dim=-1).mean(dim=0)
        vecs[i] = f.numpy()
    return vecs


def _two_groups(features):
    """KMeans into two groups. Returns (labels, silhouette 0-1, per-row boundary margin 0-1)."""
    from sklearn.metrics import silhouette_score

    x = np.asarray(features, dtype=float)
    km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(x)
    labels = km.labels_
    sizes = np.bincount(labels, minlength=2)
    sil = float(silhouette_score(x, labels)) if len(x) >= 4 and sizes.min() >= 2 else 0.0
    d = km.transform(x)
    margin = np.abs(d[:, 0] - d[:, 1]) / (d[:, 0] + d[:, 1] + 1e-6)
    return labels, max(0.0, sil), margin, km.cluster_centers_


def assign_teams(tracks, profile=None, use_embeddings=False):
    """Split tracked people into side A and B and say how sure we are.

    Net sports (tennis, volleyball) use which half of the frame a person stands in, because kits are unreliable
    there. Everything else uses kit appearance: torso colour and shorts colour, optionally cross-checked
    against CLIP appearance embeddings. Returns (mapping, team_colors, info) where info holds
    quality (0-1), method, and a per-player confidence (0 = on the boundary, 1 = clear cut)."""
    profile = profile or sports.SPORTS["soccer"]
    ids = [i for i, t in tracks.items() if len(t["pts"]) >= 4 and t["colors"]]
    by_side = profile["team_split"] == "side"
    info = {"quality": 0.0, "method": "none", "conf": {}}
    if len(ids) < (2 if by_side or profile["max_players"] <= 4 else 4):
        return {i: "A" for i in tracks}, {"A": "#ff6a2b", "B": "#2b57ff"}, info
    torso = {i: np.mean(tracks[i]["colors"], axis=0) for i in ids}
    if by_side:
        a = 1 if profile["depth_axis"] == "x" else 2          # index into (t, x, y)
        pos = {i: float(np.mean([p[a] for p in tracks[i]["pts"]])) for i in ids}
        cut = float(np.median(list(pos.values())))
        mapping = {i: ("A" if pos[i] <= cut else "B") for i in ids}
        info.update(quality=1.0, method="side", conf={i: 1.0 for i in ids})
    else:
        has_shorts = all(tracks[i].get("colors2") for i in ids)
        rows = []
        for i in ids:
            t = torso[i]
            row = [0.5 * t[0], t[1], t[2]]                    # lightness counts half: shadows vary it most
            if has_shorts:
                s = np.mean(tracks[i]["colors2"], axis=0)
                row += [0.3 * s[0], 0.6 * s[1], 0.6 * s[2]]
            rows.append(row)
        labels, sil_c, margin_c, cent_c = _two_groups(rows)
        q_c = min(1.0, sil_c / 0.5)
        method, chosen, margin, quality = "colour", labels, margin_c, q_c
        used = np.asarray(rows, dtype=float)
        if use_embeddings:
            try:
                emb = _embed_tracks(tracks, ids)
                if len(emb) == len(ids):
                    from sklearn.decomposition import PCA
                    emb_feats = np.array([emb[i] for i in ids])
                    emb_feats = PCA(n_components=min(5, len(ids) - 1)).fit_transform(emb_feats)
                    labels_e, sil_e, margin_e, _ = _two_groups(emb_feats)
                    q_e = min(1.0, sil_e / 0.5)
                    match = float(np.mean(labels == labels_e))
                    agree = abs(match - 0.5) * 2              # 0 = unrelated, 1 = same split either way round
                    if q_c < 0.45 and q_e > q_c + 0.1:        # colour is unconvincing and appearance is better
                        method, chosen, margin, quality, used = "appearance", labels_e, margin_e, q_e, emb_feats
                    quality = 0.7 * quality + 0.3 * agree
            except Exception as e:                            # never let the extra check break an analysis
                print("Appearance check skipped:", e)
        if method == "colour":
            # Vote per sample, not per average: a player whose mean sits near the line but who is clearly on one
            # side in most frames goes with the majority, so a few bad frames (shadow, occlusion) cannot flip him.
            c3, voted = cent_c[:, :3], chosen.copy()
            for j, tid in enumerate(ids):
                samp = np.asarray(tracks[tid]["colors"], dtype=float)
                if len(samp) >= 6:
                    f3 = np.c_[0.5 * samp[:, 0], samp[:, 1], samp[:, 2]]
                    share1 = float(np.mean(np.linalg.norm(f3 - c3[1], axis=1) < np.linalg.norm(f3 - c3[0], axis=1)))
                    if share1 > 0.6:
                        voted[j] = 1
                    elif share1 < 0.4:
                        voted[j] = 0
            chosen = voted
        sizes = np.bincount(chosen, minlength=2)
        if (profile.get("balanced") and len(ids) % 2 == 0 and len(ids) >= 6 and sizes.min() >= 1
                and abs(int(sizes[0]) - int(sizes[1])) >= 2):
            # both sides are always the same size in this sport, so a lopsided split is wrong: split along the
            # axis between the two groups at the median instead
            c0, c1 = used[chosen == 0].mean(axis=0), used[chosen == 1].mean(axis=0)
            order = np.argsort(used @ (c1 - c0))
            fixed = np.zeros(len(ids), dtype=int)
            fixed[order[len(ids) // 2:]] = 1
            chosen, method, quality = fixed, method + "+balanced", min(quality, 0.75)
            sizes = np.bincount(chosen, minlength=2)
        if len(ids) >= 6 and sizes.min() / len(ids) < 0.2:
            quality *= 0.5     # a very lopsided split (e.g. 23 v 3) is usually outliers, not two teams
        mapping = {i: ("A" if l == 0 else "B") for i, l in zip(ids, chosen)}
        info.update(quality=round(float(quality), 2), method=method,
                    conf={i: round(float(m), 2) for i, m in zip(ids, margin)})
    for i in tracks:
        mapping.setdefault(i, "A")
    team_colors = {}
    for name, default in (("A", "#ff6a2b"), ("B", "#2b57ff")):
        members = [torso[i] for i in ids if mapping[i] == name]
        team_colors[name] = _lab_hex(np.mean(members, axis=0)) if members else default
    return mapping, team_colors, info


def summarise(data, teams, conf=None):
    conf = conf or {}
    summaries = []
    for tid, tr in data["tracks"].items():
        pts = tr["pts"]
        if len(pts) < 4:
            continue
        arr = np.array(pts)
        dist = float(np.sum(np.hypot(np.diff(arr[:, 1]), np.diff(arr[:, 2]))))
        dt = float(arr[-1, 0] - arr[0, 0]) or 1.0
        summaries.append({
            "track_id": tid, "team": teams[tid], "team_conf": conf.get(tid, 1.0),
            "avg_x": round(float(arr[:, 1].mean()), 3), "avg_y": round(float(arr[:, 2].mean()), 3),
            "start": [round(float(arr[0, 1]), 3), round(float(arr[0, 2]), 3)],
            "end": [round(float(arr[-1, 1]), 3), round(float(arr[-1, 2]), 3)],
            "distance_norm": round(dist, 3), "avg_speed_norm_per_s": round(dist / dt, 3),
            "seen_seconds": round(dt, 1),
        })
    return summaries


def annotate(frame, rec, teams, team_colors):
    img = frame.copy()
    H, W = img.shape[:2]
    for tid, x1, y1, x2, y2 in rec["p"]:
        hexc = team_colors.get(teams.get(tid, "A"), "#ffffff").lstrip("#")
        bgr = (int(hexc[4:6], 16), int(hexc[2:4], 16), int(hexc[0:2], 16))
        p1, p2 = (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H))
        cv2.rectangle(img, p1, p2, bgr, 2)
        cv2.putText(img, f"#{tid}", (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if rec.get("ball"):
        cv2.circle(img, (int(rec["ball"][0] * W), int(rec["ball"][1] * H)), 7, (0, 255, 255), 2)
    return img


# ----------------------------------------------------------------- LLM analysis
SYSTEM = """You are an elite <<LABEL>> analyst and coach. You receive annotated frames from a clip
(boxes labelled #ID, colour = <<UNIT>>, yellow circle = ball) plus tracking statistics. Coordinates are normalised
0-1 in video-frame space (x right, y down). Judge only what the frames/stats support; if something is uncertain
say so. Be specific, tactical and a little nit-picky.
Be brutally honest. This is a coach's review, not encouragement: name mistakes plainly, give the reason, and never
pad with generic praise. Use the whole 0-10 scale: 5 is an average player, 7+ means clearly good, 8.5+ is rare,
and a player who did little or hurt the team's shape scores 3 or below. Most scores should land between 3 and 7.
Every player needs at least one real fault and one real nitpick (a small habit, not a repeat of the fault), each
tied to something visible or to a number in the stats. "key_moments" must contain at least four items with
timestamps in seconds: shots, passes, tackles, runs, skill moves or shape breakdowns, whatever actually happened.
Sport context: <<CONTEXT>>
What to judge: <<FOCUS>>.
Use the vocabulary of <<LABEL>>, and call "ideal_position" and "suggested_run" the position and movement a good
<<LABEL>> coach would ask for in that moment.
Return ONLY valid JSON matching this schema:
{
 "teams": {"A": {"formation": str, "style": str, "strengths": [str], "weaknesses": [str]}, "B": {...}},
 "players": [{
   "track_id": int, "team": "A"|"B", "role": str, "score": number 0-10,
   "doing_right": [str], "doing_wrong": [str], "nitpicks": [str],
   "ideal_position": {"x": 0-1, "y": 0-1, "why": str},
   "suggested_run": {"from": {"x":..,"y":..}, "to": {"x":..,"y":..}, "description": str}
 }],
 "key_moments": [{"time": seconds, "title": str, "description": str}],
 "summary": str
}
Include every player in the stats list that appears in at least one frame."""


def build_system_prompt(profile):
    return (SYSTEM.replace("<<LABEL>>", profile["label"]).replace("<<UNIT>>", profile["unit"].lower())
            .replace("<<CONTEXT>>", profile["context"]).replace("<<FOCUS>>", profile["focus"]))


def llm_analysis(data, teams, team_colors, stats, formations, progress=lambda p, m: None, profile=None):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    profile = profile or sports.SPORTS["soccer"]

    client = anthropic.Anthropic(api_key=key)
    content = []
    for t, frame, rec in data["keys"]:
        content.append({"type": "text", "text": f"Frame at t={t:.1f}s"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": _b64_jpeg(annotate(frame, rec, teams, team_colors))}})
    content.append({"type": "text", "text": json.dumps({
        "team_colours_hex": team_colors, "estimated_formations_from_tracking": formations,
        "player_stats": stats[:30]})})
    progress(70, "Analysing tactics with AI")
    msg = client.messages.create(model=MODEL, max_tokens=8000, system=build_system_prompt(profile),
                                 messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in msg.content if getattr(b, "text", None))
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0))


AI_NOTES = {   # why the written AI report is missing, in words a user can act on
    "nokey": "no API key is set",
    "credit": "the API account has no credit left",
    "error": "the AI request failed",
}


def _runs(mask, min_len=1):
    """(start, end_exclusive) of each run of consecutive True values."""
    out, start = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def _z(values):
    a = np.asarray(values, dtype=float)
    sd = float(a.std())
    return (a - a.mean()) / sd if sd > 1e-9 else np.zeros_like(a)


def movement_report(frames, stats, formations, profile, body_scale=False):
    """Measure how every player actually moved and turn it into ratings, faults, habits and key moments.
    Nothing here is guessed: each statement below comes from a number computed on the tracked positions.
    body_scale=True measures movement in body-lengths (each player's own box height) instead of screen fractions,
    so a player far from the camera is not marked down for the same real movement looking smaller."""
    team_of = {s["track_id"]: s["team"] for s in stats}
    tids = list(team_of)
    n = len(frames)
    if n < 6 or len(tids) < 2:
        return None
    sport_key = next((k for k, v in sports.SPORTS.items() if v is profile), "soccer")
    V = vocab.get(sport_key)                       # this sport's own words: a tennis clip never talks about a pitch
    unit = profile["unit"]
    by_side = profile.get("team_split") == "side"  # net sports: the opponent is across the net, not a marker
    t = [f["t"] for f in frames]
    small_sport = profile["max_players"] <= 6
    pos = {tid: {} for tid in tids}                # tid -> {frame index: (x, y)} camera-stabilised foot position
    ball = {}
    for i, f in enumerate(frames):
        cx, cy = f.get("cam") or (0.0, 0.0)
        for p in f["p"]:
            if p[0] in pos:
                pos[p[0]][i] = ((p[1] + p[3]) / 2 - cx, p[4] - cy)
        if f.get("ball") and not f.get("ball_est"):
            ball[i] = (f["ball"][0] - cx, f["ball"][1] - cy)

    heights = {tid: [] for tid in tids}            # box height per frame: the player's own yardstick
    for f in frames:
        for p in f["p"]:
            if p[0] in heights:
                heights[p[0]].append(max(1e-4, p[4] - p[2]))
    hgt = {tid: (float(np.median(v)) if v else 0.05) for tid, v in heights.items()}

    # ---- per frame: team centres, spacing to teammates and to opponents
    need = 2 if small_sport else 3
    acc = {tid: {"cent": [], "mate": [], "opp": [], "ball": []} for tid in tids}
    spread_by_frame = {}
    team_spread = {"A": [], "B": []}
    for i in range(n):
        here = [(tid, pos[tid][i]) for tid in tids if i in pos[tid]]
        if len(here) < 2:
            continue
        allc = np.mean([p for _, p in here], axis=0)
        spread_by_frame[i] = float(np.mean([math.hypot(p[0] - allc[0], p[1] - allc[1]) for _, p in here]))
        for team in ("A", "B"):
            mem = [(tid, p) for tid, p in here if team_of[tid] == team]
            if len(mem) < need:
                continue
            c = np.mean([p for _, p in mem], axis=0)
            team_spread[team].append(float(np.mean([math.hypot(p[0] - c[0], p[1] - c[1]) for _, p in mem])))
            for tid, p in mem:
                acc[tid]["cent"].append(math.hypot(p[0] - c[0], p[1] - c[1]))
                others = [q for u, q in mem if u != tid]
                if others:
                    acc[tid]["mate"].append(min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in others))
                opp = [q for u, q in here if team_of[u] != team]
                if opp:
                    acc[tid]["opp"].append(min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in opp))
        if i in ball:
            b = ball[i]
            for tid, p in here:
                acc[tid]["ball"].append(math.hypot(p[0] - b[0], p[1] - b[1]))

    # ---- movement: distance, pace, stillness, bursts, sharpest turn
    # Whatever every player does together between two frames is camera shake or drift, not play. Remove it, so
    # a jolt of the drone is not scored as 22 players suddenly turning round.
    common = {}
    for a in range(n - 1):
        for b in (a + 1, a + 2):
            if b >= n:
                continue
            both = [x for x in tids if a in pos[x] and b in pos[x]]
            if len(both) >= 5:
                common[(a, b)] = (float(np.median([pos[x][b][0] - pos[x][a][0] for x in both])),
                                  float(np.median([pos[x][b][1] - pos[x][a][1] for x in both])))
    steps = {}                                      # tid -> list of (frame_a, frame_b, speed, dx, dy)
    for tid in tids:
        idx = sorted(pos[tid])
        st = []
        for a, b in zip(idx[:-1], idx[1:]):
            dt = t[b] - t[a]
            if b - a <= 2 and dt > 0:
                cdx, cdy = common.get((a, b), (0.0, 0.0))
                dx, dy = pos[tid][b][0] - pos[tid][a][0] - cdx, pos[tid][b][1] - pos[tid][a][1] - cdy
                if body_scale:
                    dx, dy = dx / hgt[tid], dy / hgt[tid]
                st.append((a, b, math.hypot(dx, dy) / dt, dx, dy))
        steps[tid] = st
    # a tracker ID swap looks like a teleport. Nobody runs 7x the typical pace of the whole clip: drop those steps.
    raw = [s[2] for tid in tids for s in steps[tid]]
    cap = 7.0 * max(float(np.median(raw)) if raw else 1e-6, 1e-6)
    all_speeds = []
    for tid in tids:
        steps[tid] = [s for s in steps[tid] if s[2] <= cap]
        all_speeds += [s[2] for s in steps[tid]]
    med = float(np.median(all_speeds)) if all_speeds else 1e-6
    med = max(med, 1e-6)

    m = {}
    for tid in tids:
        st = steps[tid]
        sp = np.array([s[2] for s in st]) if st else np.array([0.0])
        sm = np.convolve(sp, np.ones(3) / 3, mode="same") if len(sp) >= 3 else sp
        dist = float(sum(math.hypot(s[3], s[4]) for s in st))
        dur = float(sum(t[s[1]] - t[s[0]] for s in st)) or 1.0
        runs = _runs(sm > 1.8 * med, min_len=3)
        peak_i = int(np.argmax(sm)) if len(sm) else 0
        xs = np.array([p[0] for p in pos[tid].values()]) if pos[tid] else np.array([0.0])
        ys = np.array([p[1] for p in pos[tid].values()]) if pos[tid] else np.array([0.0])
        turn, turn_t = 0.0, None
        for k in range(len(st) - 1):
            a, b = st[k], st[k + 1]
            if b[0] == a[1] and a[2] > med and b[2] > med and t[a[0]] - t[0] > 0.5:   # not the first frames: box jitter
                va, vb = (a[3], a[4]), (b[3], b[4])
                na, nb = math.hypot(*va), math.hypot(*vb)
                if na > (0.02 if body_scale else 0.004) and nb > (0.02 if body_scale else 0.004):
                    ang = math.degrees(math.acos(max(-1.0, min(1.0, (va[0] * vb[0] + va[1] * vb[1]) / (na * nb)))))
                    if ang > turn:
                        turn, turn_t = ang, t[b[0]]
        bd = acc[tid]["ball"]
        m[tid] = {
            "dist": dist, "pace": dist / dur, "peak": float(np.percentile(sm, 90)) if len(sm) else 0.0,
            "peak_t": (t[st[peak_i][0]] + t[st[peak_i][1]]) / 2 if st else t[0],
            "still": float(np.mean(sp < 0.35 * med)), "sprints": len(runs),
            "zone": float(math.sqrt(xs.var() + ys.var())) / (hgt[tid] if body_scale else 1.0),
            "cent": float(np.mean(acc[tid]["cent"])) if acc[tid]["cent"] else None,
            "mate": float(np.mean(acc[tid]["mate"])) if acc[tid]["mate"] else None,
            "opp": float(np.mean(acc[tid]["opp"])) if acc[tid]["opp"] else None,
            "ball": float(np.mean(bd)) if len(bd) >= 5 else None,
            "near": float(np.mean(np.array(bd) < 0.08)) if len(bd) >= 5 else None,
            "turn": turn, "turn_t": turn_t,
            "seen": len(pos[tid]) / n,              # share of the clip's frames this player is actually tracked in
        }

    # ---- scores: rank every player against everyone else in the clip (5 = average, brutal at both ends)
    # ball-based claims ("rarely near the ball") need the ball to have been seen often enough to mean something
    have_ball = (sum(1 for tid in tids if m[tid]["ball"] is not None) >= max(3, 0.6 * len(tids))
                 and len(ball) >= 0.2 * n)
    zs = {
        "work": _z([m[x]["pace"] for x in tids]),
        "burst": _z([m[x]["sprints"] + 3 * m[x]["peak"] / med for x in tids]),
        "shape": -_z([m[x]["cent"] if m[x]["cent"] is not None else 0 for x in tids]),
        "engage": (-_z([m[x]["ball"] for x in tids]) if have_ball
                   else -_z([m[x]["opp"] if m[x]["opp"] is not None else 0 for x in tids])),
    }
    weights = {"work": 0.35, "burst": 0.15, "shape": 0.20, "engage": 0.30}
    mean_of = lambda k: float(np.mean([m[x][k] for x in tids if m[x][k] is not None] or [0.0]))
    avg = {k: mean_of(k) for k in ("pace", "cent", "mate", "opp", "ball", "near", "still", "zone")}
    cx_all = float(np.mean([s["avg_x"] for s in stats])); cy_all = float(np.mean([s["avg_y"] for s in stats]))
    depth_is_x = profile["depth_axis"] == "x"
    pct = lambda v: f"{abs(v) * 100:.0f}%"
    dax = 0 if depth_is_x else 1
    side_mean = {tm: float(np.mean([(x["avg_x"], x["avg_y"])[dax] for x in stats if x["team"] == tm] or [0.5]))
                 for tm in ("A", "B")}
    net_line = (side_mean["A"] + side_mean["B"]) / 2
    med_net = float(np.median([abs((x["avg_x"], x["avg_y"])[dax] - net_line) for x in stats]))
    d_all = {x["track_id"]: math.hypot(x["avg_x"] - cx_all, x["avg_y"] - cy_all) for x in stats}
    med_d = float(np.median(list(d_all.values())))
    players = []
    for k, tid in enumerate(tids):
        s = next(x for x in stats if x["track_id"] == tid)
        mm = m[tid]
        raw = sum(weights[w] * float(zs[w][k]) for w in weights)
        score = float(np.clip(5.0 + 2.6 * raw, 1.0, 9.6))      # harsh on purpose: 5 is average, 2-3 is poor, 8+ is rare
        # a player tracked for only part of the clip cannot be rated with full confidence: pull toward 5 a little,
        # in proportion to how little of the clip he was seen (smooth, so no ties and no cliff)
        score = 5.0 + (score - 5.0) * float(np.clip(0.55 + mm["seen"] * 1.2, 0.55, 1.0))
        right, wrong, nits = [], [], []
        work_ratio = mm["pace"] / avg["pace"] if avg["pace"] else 1.0
        if zs["work"][k] > 0.5:
            right.append(V["work_hi"].format(p=pct(work_ratio - 1)))
        elif zs["work"][k] < -0.5:
            wrong.append(V["work_lo"].format(p=pct(1 - work_ratio)))
        if mm["sprints"] >= 1 and zs["burst"][k] > 0.4:
            right.append(V["burst"].format(n=mm["sprints"], s="s" if mm["sprints"] > 1 else "", r=mm["peak"] / med))
        if mm["still"] > max(0.45, avg["still"] * 1.4):
            wrong.append(V["still"].format(p=f"{mm['still'] * 100:.0f}%"))
        if mm["cent"] is not None and avg["cent"]:
            ratio = mm["cent"] / avg["cent"]
            if zs["shape"][k] > 0.5:
                right.append(V["shape_hi"].format(p=pct(1 - ratio)))
            elif zs["shape"][k] < -0.5:
                wrong.append(V["shape_lo"].format(r=ratio))
        if not by_side and mm["opp"] is not None and avg["opp"]:
            r = mm["opp"] / avg["opp"]
            if r < 0.75:
                right.append(V["opp_hi"].format(p=pct(1 - r)))
            elif r > 1.35:
                wrong.append(V["opp_lo"].format(r=r))
        if have_ball and mm["near"] is not None:
            if mm["near"] > max(0.12, avg["near"] * 1.5):
                right.append(V["ball_hi"].format(p=f"{mm['near'] * 100:.0f}%"))
            elif mm["near"] < avg["near"] * 0.4:
                wrong.append(V["ball_lo"].format(p=f"{mm['near'] * 100:.0f}%"))
        if mm["mate"] is not None and avg["mate"]:
            r = mm["mate"] / avg["mate"]
            if r > 1.5:
                nits.append(V["mate_far"].format(r=r))
            elif r < 0.6:
                nits.append(V["mate_near"].format(p=pct(1 - r)))
        if mm["zone"] < avg["zone"] * 0.55:
            nits.append(V["zone_small"])
        elif mm["zone"] > avg["zone"] * 1.6 and not zs["shape"][k] > 0.5:   # not alongside "stayed connected"
            nits.append(V["zone_big"])
        if mm["turn"] > 120:
            nits.append(V["turn"].format(a=mm["turn"], t=mm["turn_t"]))
        if mm["peak"] and mm["pace"] and mm["peak"] / mm["pace"] < 1.5 and zs["work"][k] > 0:
            nits.append(V["pace_flat"])
        if mm["seen"] < 0.5:
            nits.append(V["seen_low"].format(p=f"{mm['seen'] * 100:.0f}%"))
        if (s.get("team_conf", 1.0) or 1.0) < 0.15:
            nits.append(V["team_unsure"])
        if not right:
            right.append(V["none_right"])
        if not wrong:
            wrong.append(V["none_wrong"])
        if not nits:
            nits.append(V["none_nit"])

        # where he should have been: pulled toward his team's shape, and toward the ball side when it was seen
        team_pts = [(x["avg_x"], x["avg_y"]) for x in stats if x["team"] == s["team"]]
        tcx, tcy = float(np.mean([p[0] for p in team_pts])), float(np.mean([p[1] for p in team_pts]))
        pull = float(np.clip(0.15 + 0.22 * max(0.0, -float(zs["shape"][k])), 0.1, 0.5))
        ix, iy = s["avg_x"] + (tcx - s["avg_x"]) * pull, s["avg_y"] + (tcy - s["avg_y"]) * pull
        if ball:
            bx = float(np.mean([b[0] for b in ball.values()])); by = float(np.mean([b[1] for b in ball.values()]))
            ix, iy = ix + (bx - ix) * 0.12, iy + (by - iy) * 0.12
        single = len(team_pts) == 1                # one player on this side (tennis singles): no team centre to use
        if single:
            ix, iy = 0.5 * s["avg_x"] + 0.5 * cx_all, s["avg_y"]       # recover toward the middle line, hold depth
        ix, iy = float(np.clip(ix, 0.03, 0.97)), float(np.clip(iy, 0.03, 0.97))
        gap = math.hypot(ix - s["avg_x"], iy - s["avg_y"])
        if single:
            why = V["why_ok"] if gap < 0.02 else V["why_single"]
        elif zs["shape"][k] < -0.5:
            why = V["why_far"].format(r=mm["cent"] / avg["cent"], p=f"{pull * 100:.0f}%")
        elif gap < 0.02:
            why = V["why_ok"]
        else:
            why = V["why_adj"].format(ballside=V["ballside"] if ball else "")
        # a label from where they stood across the clip, in the sport's words
        kind = V["role_kind"]
        if kind == "court":                        # tennis: which end of the court
            role = f"{V['role_near'] if s['avg_y'] >= cy_all else V['role_far']} {V['role_noun']}"
        elif kind == "frontback":                  # volleyball: distance from the net
            net = (side_mean["A"] + side_mean["B"]) / 2
            own = abs((s["avg_x"] if depth_is_x else s["avg_y"]) - net)
            role = f"{V['role_front'] if own <= med_net else V['role_back']} {V['role_noun']}"
        elif kind == "ring":                       # baseball / cricket: inner or outer part of the field
            role = f"{V['role_inner'] if d_all[tid] <= med_d else V['role_outer']}"
        else:
            perp = (s["avg_y"] - tcy) if depth_is_x else (s["avg_x"] - tcx)
            width = V["role_wide"] if abs(perp) > 0.18 else V["role_central"]
            tempo = "roaming" if zs["work"][k] > 0.8 else "holding" if zs["work"][k] < -0.8 else ""
            role = " ".join(w for w in (width, tempo, V["role_noun"]) if w)
        players.append({
            "track_id": tid, "team": s["team"], "role": role, "score": round(score, 1),
            "doing_right": right[:3], "doing_wrong": wrong[:3], "nitpicks": nits[:3],
            "ideal_position": {"x": round(ix, 3), "y": round(iy, 3), "why": why},
            "suggested_run": {"from": {"x": s["end"][0], "y": s["end"][1]}, "to": {"x": round(ix, 3), "y": round(iy, 3)},
                              "description": V["run"].format(p=f"{gap * 100:.0f}%")},
        })

    # ---- team write-ups
    teams_out = {}
    ts = {k: (float(np.mean(v)) if v else None) for k, v in team_spread.items()}
    tp = {k: float(np.mean([m[x]["pace"] for x in tids if team_of[x] == k] or [0.0])) for k in ("A", "B")}
    for k, other in (("A", "B"), ("B", "A")):
        strengths, weaknesses = [], []
        tight_w, wide_w, even_w = V["shape_words"]
        if ts[k] and ts[other]:
            if V["wide_is_good"]:                       # basketball: spacing is the goal, so wider is the strength
                if ts[k] > ts[other] * 1.1:
                    strengths.append(V["team_wide_good"].format(p=pct(ts[k] / ts[other] - 1)))
                elif ts[k] < ts[other] * 0.9:
                    weaknesses.append(V["team_tight_bad"].format(p=pct(1 - ts[k] / ts[other])))
            else:
                if ts[k] < ts[other] * 0.9:
                    strengths.append(V["team_tight_good"].format(p=pct(1 - ts[k] / ts[other])))
                elif ts[k] > ts[other] * 1.1:
                    weaknesses.append(V["team_wide_bad"].format(p=pct(ts[k] / ts[other] - 1)))
        if tp[other] > 0:
            if tp[k] > tp[other] * 1.1:
                strengths.append(V["team_pace_good"].format(p=pct(tp[k] / tp[other] - 1)))
            elif tp[k] < tp[other] * 0.9:
                weaknesses.append(V["team_pace_bad"].format(p=pct(1 - tp[k] / tp[other])))
        low = [p for p in players if p["team"] == k and p["score"] < 4]
        if low:
            weaknesses.append(V["team_low"].format(n=len(low), s="s" if len(low) > 1 else "",
                                                   ids=", ".join(f"#{p['track_id']}" for p in low[:4])))
        top = [p for p in players if p["team"] == k and p["score"] >= 6.5]
        if top:
            strengths.append(V["team_top"].format(ids=", ".join(f"#{p['track_id']} ({p['score']})" for p in
                                                                sorted(top, key=lambda p: -p["score"])[:3])))
        tempo = "high-tempo" if tp[k] > tp[other] * 1.1 else "low-tempo" if tp[k] < tp[other] * 0.9 else "even-tempo"
        if ts[k] and ts[other]:
            shape = (tight_w if ts[k] < ts[other] * 0.9 else wide_w if ts[k] > ts[other] * 1.1 else even_w)
            style = f"{shape.capitalize()}, {tempo}"
        else:                                          # nothing to compare spacing with (tennis singles): use the set-up
            style = f"{formations.get(k, 'unclear')}, {tempo}"
        teams_out[k] = {"formation": formations.get(k, "unclear"), "style": style,
                        "strengths": strengths or [V["team_none_good"]],
                        "weaknesses": weaknesses or [V["team_none_bad"]]}

    # ---- key moments (always some, because every clip has a fastest burst and a loosest moment)
    moments = []

    def add(time, title, desc):
        if all(abs(time - x["time"]) >= 0.8 for x in moments):
            moments.append({"time": round(float(time), 1), "title": title, "description": desc})

    best_sprint = max(tids, key=lambda x: m[x]["peak"])
    add(m[best_sprint]["peak_t"], V["m_sprint_t"],
        V["m_sprint_d"].format(id=best_sprint, u=unit, team=team_of[best_sprint], r=m[best_sprint]["peak"] / med))
    if len(ball) >= 6:
        bi = sorted(ball)
        bs = [(math.hypot(ball[b][0] - ball[a][0], ball[b][1] - ball[a][1]) / (t[b] - t[a]), a, b)
              for a, b in zip(bi[:-1], bi[1:]) if b - a <= 3 and t[b] > t[a]]
        if bs:
            sp, a, b = max(bs)
            typical = float(np.median([x[0] for x in bs])) or 1e-6
            if sp > 2.5 * typical:
                add((t[a] + t[b]) / 2, V["m_ball_t"], V["m_ball_d"].format(r=sp / typical))
    if spread_by_frame:
        ks = sorted(spread_by_frame)
        sm_sp = np.convolve([spread_by_frame[i] for i in ks], np.ones(5) / 5, mode="same")
        lo, hi = ks[int(np.argmin(sm_sp[2:-2])) + 2] if len(ks) > 6 else ks[0], ks[int(np.argmax(sm_sp[2:-2])) + 2] if len(ks) > 6 else ks[-1]
        add(t[lo], V["m_squeeze_t"], V["m_squeeze_d"])
        add(t[hi], V["m_stretch_t"], V["m_stretch_d"])
    sharp = max(tids, key=lambda x: m[x]["turn"])
    if m[sharp]["turn"] > 110 and m[sharp]["turn_t"] is not None:
        add(m[sharp]["turn_t"], V["m_turn_t"], V["m_turn_d"].format(id=sharp, a=m[sharp]["turn"]))
    add(t[0], V["m_start_t"], V["m_start_d"].format(u=unit, fa=formations.get("A", "unclear"), fb=formations.get("B", "unclear")))
    last_k = max(len(f["p"]) for f in frames[-5:])
    if last_k and t[-1] - t[0] > 3:                # a closing marker, so even a quiet clip has start / middle / end
        add(t[-1], V["m_end_t"],
            V["m_end_d"].format(k=last_k) if last_k >= 0.5 * len(tids) else V["m_end_generic"])
    moments.sort(key=lambda x: x["time"])

    ranked = sorted(players, key=lambda p: -p["score"])
    best, worst = ranked[0], ranked[-1]
    summary = (f"Rated from {n} tracked frames: #{best['track_id']} ({unit} {best['team']}) is the best performer on "
               f"{best['score']}, and #{worst['track_id']} ({unit} {worst['team']}) the weakest on {worst['score']}. ")
    if ts["A"] and ts["B"]:
        tighter = "A" if ts["A"] < ts["B"] else "B"
        summary += V["summary_tight"].format(u=unit, t=tighter, p=pct(1 - min(ts["A"], ts["B"]) / max(ts["A"], ts["B"])))
    summary += V["summary_ball_yes"] if have_ball else V["summary_ball_no"]
    return {"players": players, "teams": teams_out, "key_moments": moments, "summary": summary}


def heuristic_analysis(stats, formations, profile=None, why="nokey", frames=None, body_scale=False):
    """Data-driven report used when the written AI report is not available. Every rating, fault and key
    moment comes from the tracked movement, so it is specific to the clip and not boilerplate."""
    profile = profile or sports.SPORTS["soccer"]
    rep = movement_report(frames or [], stats, formations, profile, body_scale)
    if rep is not None:
        return rep
    # too little data for a movement report (a very short clip): say so plainly
    by_team = {"A": [s for s in stats if s["team"] == "A"], "B": [s for s in stats if s["team"] == "B"]}
    players = []
    for team, lst in by_team.items():
        for s in lst:
            players.append({
                "track_id": s["track_id"], "team": team, "role": profile["role"], "score": 5.0,
                "doing_right": ["Too little footage to credit anything."],
                "doing_wrong": ["Too little footage to fault anything."],
                "nitpicks": [f"Only {s['seen_seconds']}s on screen: a longer clip gives a fair rating."],
                "ideal_position": {"x": s["avg_x"], "y": s["avg_y"], "why": "Not enough movement to suggest a change."},
                "suggested_run": {"from": {"x": s["end"][0], "y": s["end"][1]}, "to": {"x": s["avg_x"], "y": s["avg_y"]},
                                  "description": "Not enough movement to suggest a run."},
            })
    return {"teams": {k: {"formation": formations.get(k, "unclear"), "style": "Too little footage",
                          "strengths": [], "weaknesses": []} for k in ("A", "B")},
            "players": players, "key_moments": [],
            "summary": "This clip is too short for a fair rating. Try 10 seconds or more of continuous play."}


# ----------------------------------------------------------------- entry point
def analyse(path, progress=lambda p, m: None, sport="soccer", deep=True):
    """Full analysis. deep=True is the slow, thorough scan (more frames per second, higher resolution,
    a dedicated ball pass, track stitching, appearance check); deep=False is the quick scan."""
    sport, profile = sports.get(sport)
    detector = choose_detector(path, profile, progress)
    if deep and detector["kind"] == "yolo":     # deep scan always looks at full resolution with a stronger model
        weights = "yolov8s.pt" if detector["weights"] == YOLO_WEIGHTS else detector["weights"]
        imgsz = max(1280, detector["imgsz"])
        detector = {"kind": "yolo", "weights": weights, "imgsz": imgsz, "name": f"{weights[:-3]}@{imgsz}"}
    data = track_video(path, progress, detector, deep)
    progress(58, "Linking fragmented tracks")
    progress(59, "Finding the playing area")
    # Before stitching: joining fragments inside a packed row of lookalike spectators would invent "players"
    # that seem to run the length of the touchline.
    area = fieldmask.apply(data, profile) if detector["kind"] == "yolo" else {"applied": False, "ignored_people": 0}
    merged = stitch_tracks(data)               # joins fragments of one player; cheap, so quick scans get it too
    ball = finalize_ball(data)
    selection = select_players(data, profile, detector["kind"] == "yolo")
    data.pop("thumbs", None)
    filled = densify_tracks(data)              # short detection gaps are the same person: no blinking markers
    progress(62, "Working out the teams")
    teams, team_colors, team_info = assign_teams(data["tracks"], profile, use_embeddings=deep)
    stats = summarise(data, teams, team_info["conf"])
    if not stats:
        raise RuntimeError("No players were found in this clip. Try a wider shot where the players are "
                           "clearly visible and the camera stays on the action for a few seconds.")
    formations = {}
    for t in ("A", "B"):
        pts = [(s["avg_x"], s["avg_y"]) for s in stats if s["team"] == t]
        formations[t] = estimate_formation(pts, profile) if pts else "unclear"
    why = None            # set when the written AI report could not be produced
    try:
        analysis = llm_analysis(data, teams, team_colors, stats, formations, progress, profile)
        if analysis is None:
            why = "nokey"
    except Exception as e:  # keep the demo alive if the API call fails
        analysis = None
        why = "credit" if "credit balance" in str(e).lower() else "error"
        print("LLM analysis failed:", e)
    used_ai = analysis is not None
    if analysis is None:
        analysis = heuristic_analysis(stats, formations, profile, why, data["frames"], detector["kind"] == "yolo")
    elif len(analysis.get("key_moments") or []) < 3:
        # an AI report with too few moments gets the measured ones added, so every clip has things to jump to
        extra = (movement_report(data["frames"], stats, formations, profile, detector["kind"] == "yolo") or {}).get("key_moments", [])
        have = analysis.get("key_moments") or []
        analysis["key_moments"] = sorted(have + [e for e in extra if all(abs(e["time"] - h.get("time", -9)) > 0.8 for h in have)],
                                         key=lambda x: x["time"])
    for f in data["frames"]:
        f.pop("cam", None)
        f.pop("M", None)
    progress(95, "Finishing")
    return _plain({
        "sport": sport,
        "video": {"fps": data["fps"], "width": data["W"], "height": data["H"]},
        "team_colors": team_colors, "formations": formations,
        "frames": data["frames"], "stats": stats, "analysis": analysis,
        "mode": "ai" if used_ai else "heuristic",
        "ai_note": why,
        "detector": detector.get("name", "yolo"),
        "scan": {
            "deep": bool(deep), "fps": round(float(data.get("step_fps", 0)), 1), "imgsz": detector.get("imgsz"),
            "frames": len(data["frames"]), "players": len(stats), "merged_tracks": merged,
            "candidates": selection["candidates"], "fragments_joined": selection["merged_fragments"],
            "far_removed": selection["far_removed"],
            "area_applied": bool(area.get("applied")), "ignored_people": area.get("ignored_people", 0),
            "gaps_filled": filled,
            "ball_seen": ball["seen"], "ball_filled": ball["filled"],
            "team_quality": team_info["quality"], "team_method": team_info["method"],
        },
    })


def select_players(data, profile, yolo=False):
    """Decide who the players actually are, then tidy the tracks that describe them.

    1. Score every track: how much of the clip it is present, how much it moves, how close it is to the action.
    2. Choose the player count INSIDE the sport's plausible range (a tennis clip has 2 or 4 players, never 3)
       by finding where the scores genuinely fall off a cliff: players on one side, strays on the other.
    3. Glue stray fragments back onto the player they belong to (the tracker re-labels people it loses).
    4. Drop anyone far from the main group: that is another game, a spectator or a coach.
    Returns counts describing what was done, so the report can show its working."""
    frames, tracks = data["frames"], data["tracks"]
    n = max(1, len(frames))
    lo, hi = profile.get("count_range", (2, profile["max_players"]))
    hi = max(lo, min(hi, profile["max_players"]))

    occ = {}                                            # track id -> set of frame indices it appears in
    for i, f in enumerate(frames):
        for p in f["p"]:
            occ.setdefault(p[0], set()).add(i)
    info = {}
    for tid, tr in tracks.items():
        if len(tr["pts"]) < 4:
            continue
        arr = np.asarray(tr["pts"], dtype=float)
        info[tid] = {"pres": len(arr) / n, "x": float(arr[:, 1].mean()), "y": float(arr[:, 2].mean()),
                     "path": float(np.sum(np.hypot(np.diff(arr[:, 1]), np.diff(arr[:, 2]))))}
    stats_out = {"candidates": len(info), "kept": 0, "merged_fragments": 0, "far_removed": 0, "count_chosen": 0}
    if not info:
        data["tracks"] = {}
        for rec in frames:
            rec["p"] = []
        return stats_out

    # ---- 1. score
    ids = list(info)
    ref = float(np.percentile([info[i]["path"] for i in ids], 75)) or 1e-6
    body = fieldmask.track_stats(data) if yolo else {}      # distance travelled in body-lengths, camera removed
    for i in ids:
        if i in body:
            mot = 0.35 + 0.65 * min(1.0, body[i]["ext"] / 2.5)       # a statue (coach, official) is not a player
        else:
            mot = 0.4 + 0.6 * min(1.0, info[i]["path"] / (0.6 * ref + 1e-9))
        info[i]["mot"] = mot
        info[i]["w"] = info[i]["pres"] * info[i]["mot"]
    top = sorted(ids, key=lambda i: -info[i]["w"])[:max(2, min(hi, 6))]
    wsum = sum(info[i]["w"] for i in top) or 1.0
    cx = sum(info[i]["x"] * info[i]["w"] for i in top) / wsum
    cy = sum(info[i]["y"] * info[i]["w"] for i in top) / wsum
    scale = max(0.3, 2.4 * float(np.median([math.hypot(info[i]["x"] - cx, info[i]["y"] - cy) for i in top])))
    for i in ids:
        d = math.hypot(info[i]["x"] - cx, info[i]["y"] - cy)
        info[i]["score"] = (info[i]["pres"] ** 0.8) * info[i]["mot"] * (0.35 + 0.65 * math.exp(-(d / scale) ** 2))
    ranked = sorted(ids, key=lambda i: -info[i]["score"])
    s = [info[i]["score"] for i in ranked]
    m = len(ranked)

    # ---- 2. how many players? the biggest genuine drop in score, inside the allowed range
    lo_k, hi_k = min(lo, m), min(hi, m)
    k, best = hi_k, -1.0
    if hi_k > lo_k:
        for kk in range(lo_k, hi_k + 1):
            drop = (s[kk - 1] - s[kk]) / max(s[kk - 1], 1e-9) if kk < m else 0.0   # no next track = no evidence
            val = drop * (1.0 + 0.2 * (kk - lo_k) / max(1, hi_k - lo_k))            # lean slightly to the bigger count
            if val > best:
                best, k = val, kk
        if best < 0.25:                                  # no clear cliff: keep everything that has real presence
            k = hi_k
    anchors = [i for i in ranked[:k] if info[i]["score"] >= 0.2 * s[0] and info[i]["pres"] >= 0.05] or ranked[:1]
    stats_out["count_chosen"] = len(anchors)
    stats_out["ranking"] = [(i, round(info[i]["score"], 3), round(info[i]["pres"], 2), round(info[i]["mot"], 2),
                             round(info[i]["x"], 2), round(info[i]["y"], 2)) for i in ranked[:40]]

    # ---- 3. fragments go back to the player they belong to, or are discarded as noise
    timeline = {a: np.asarray(sorted(tracks[a]["pts"]), dtype=float) for a in anchors}
    colour = {i: (np.mean(tracks[i]["colors"], axis=0) if tracks[i]["colors"] else None) for i in ids}
    merged_into = {}
    for fid in [i for i in ranked if i not in set(anchors)]:
        fpts = np.asarray(tracks[fid]["pts"], dtype=float)
        best_a, best_cost = None, 1e9
        for a in anchors:
            if occ.get(a, set()) & occ.get(fid, set()):
                continue                                 # two boxes in one frame means two different people
            at = timeline[a]
            j = np.abs(at[:, 0][None, :] - fpts[:, 0][:, None]).argmin(axis=1)
            gap = float(np.median(np.abs(at[j, 0] - fpts[:, 0])))
            dpos = float(np.mean(np.hypot(at[j, 1] - fpts[:, 1], at[j, 2] - fpts[:, 2])))
            ca, cf = colour[a], colour[fid]
            cdist = float(np.linalg.norm(ca - cf)) if ca is not None and cf is not None else 0.0
            if gap > 10 or dpos > 0.25 or cdist > 45:
                continue
            cost = dpos + 0.03 * max(0.0, gap - 1.0) + cdist / 400.0
            if cost < best_cost:
                best_a, best_cost = a, cost
        if best_a is not None and best_cost < 0.4:
            merged_into[fid] = best_a
            occ[best_a] = occ.get(best_a, set()) | occ.get(fid, set())
            timeline[best_a] = np.vstack([timeline[best_a], fpts])
            timeline[best_a] = timeline[best_a][np.argsort(timeline[best_a][:, 0])]
    stats_out["merged_fragments"] = len(merged_into)

    # ---- 4. far from the main group = another game or a bystander
    pres_w = {a: max(len(timeline[a]), 1) for a in anchors}
    gx = sum(float(timeline[a][:, 1].mean()) * pres_w[a] for a in anchors) / sum(pres_w.values())
    gy = sum(float(timeline[a][:, 2].mean()) * pres_w[a] for a in anchors) / sum(pres_w.values())
    dist = {a: math.hypot(float(timeline[a][:, 1].mean()) - gx, float(timeline[a][:, 2].mean()) - gy) for a in anchors}
    limit = max(0.32, 3.0 * float(np.median(list(dist.values()))))
    far = [a for a in anchors if dist[a] > limit]
    if far and len(anchors) - len(far) >= min(lo, len(anchors)):
        anchors = [a for a in anchors if a not in far]
        merged_into = {f: a for f, a in merged_into.items() if a in anchors}
        stats_out["far_removed"] = len(far)

    # ---- rebuild: renumber 1..N (1 = on screen longest), one box per player per frame
    anchors.sort(key=lambda a: -len(timeline[a]))
    new_id = {a: i for i, a in enumerate(anchors, start=1)}
    for f, a in merged_into.items():
        new_id[f] = new_id[a]
    new_tracks = {}
    for a in anchors:
        members = [a] + [f for f, o in merged_into.items() if o == a]
        pts = sorted(p for mid in members for p in tracks[mid]["pts"])
        new_tracks[new_id[a]] = {
            "pts": pts,
            "colors": [c for mid in members for c in tracks[mid]["colors"]],
            "colors2": [c for mid in members for c in tracks[mid].get("colors2", [])],
            "crops": [c for mid in members for c in tracks[mid].get("crops", [])][:6],
        }
    seen = set()                                         # keyframe records ARE frame records: remap each once
    for rec in list(frames) + [r for _, _, r in data["keys"]]:
        if id(rec) in seen:
            continue
        seen.add(id(rec))
        out, used = [], set()
        for p in rec["p"]:
            nid = new_id.get(p[0])
            if nid is None or nid in used:
                continue
            used.add(nid)
            out.append([nid, *p[1:]])
        rec["p"] = out
    data["tracks"] = new_tracks
    stats_out["kept"] = len(new_tracks)
    return stats_out


def densify_tracks(data, max_gap_s=1.6):
    """Make every kept player's ring and number stay on screen.

    The detector misses a player for a frame or two (a block, motion blur, someone crossing in front) and
    the overlay would drop their marker and bring it back, which looks like blinking. A gap that short is
    the same person who is still there, so the box is filled in between the two sightings. Then the box
    coordinates are lightly smoothed so markers do not jitter. A genuine absence (out of shot for longer than
    max_gap_s) is left as a gap. Returns how many boxes were filled in."""
    frames = data["frames"]
    if len(frames) < 3:
        return 0
    seen = {}                                            # id -> {frame index: the [id, x1, y1, x2, y2] list}
    for i, rec in enumerate(frames):
        for p in rec["p"]:
            seen.setdefault(p[0], {})[i] = p
    filled = 0
    for tid, obs in seen.items():
        idx = sorted(obs)
        for a, b in zip(idx, idx[1:]):
            ta, tb = frames[a]["t"], frames[b]["t"]
            if b - a <= 1 or tb - ta > max_gap_s:
                continue
            pa, pb = obs[a], obs[b]
            for k in range(a + 1, b):
                u = (frames[k]["t"] - ta) / max(1e-6, tb - ta)
                box = [tid] + [round(pa[j] + u * (pb[j] - pa[j]), 4) for j in range(1, 5)]
                frames[k]["p"].append(box)
                obs[k] = box
                filled += 1
                tr = data["tracks"].get(tid)
                if tr is not None:                       # keep the track's own point list in step
                    tr["pts"].append((frames[k]["t"], (box[1] + box[3]) / 2, box[4]))
    for tr in data["tracks"].values():
        tr["pts"].sort(key=lambda q: q[0])
    for tid, obs in seen.items():                        # 3-tap smoothing inside each unbroken run
        idx = sorted(obs)
        run = []
        for i in idx + [None]:
            if run and (i is None or i != run[-1] + 1):
                if len(run) >= 3:
                    c = np.array([obs[j][1:5] for j in run], dtype=float)
                    s = c.copy()
                    s[1:-1] = 0.25 * c[:-2] + 0.5 * c[1:-1] + 0.25 * c[2:]
                    for j, row in zip(run, s):
                        obs[j][1:5] = [round(float(v), 4) for v in row]
                run = []
            if i is not None:
                run.append(i)
    return filled


def _plain(o):
    """Convert NumPy numbers/arrays (float32, int64, ...) to plain Python so the result is JSON-safe."""
    if isinstance(o, dict):
        return {(k.item() if hasattr(k, "item") else k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "tolist"):  # numpy arrays and numpy scalars
        return o.tolist()
    return o
