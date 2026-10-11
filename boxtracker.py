"""A small multi-person tracker built for sport footage where people are small and bunched together.

Why not the tracker that ships with YOLO (ByteTrack)? It only reports a person once it has matched them in two
frames in a row by box overlap. A 25 px-wide player who moves 15 px between analysed frames has almost no overlap
with their previous box, so they are never "confirmed" and silently vanish: in a wide shot of a football
line-up the tracker reported 2-5 players per frame while the detector itself saw 12.

This one matches by distance measured in body-lengths (so it works at any zoom), does it in camera-stabilised
coordinates (so a panning phone does not make everyone look like they are sprinting), adds a kit-colour check so
teams do not swap, and puts a box on every confident detection straight away.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e6
GATE_BASE, GATE_PER_FRAME = 0.55, 0.20    # how far (body-lengths) a person may move between analysed frames
RATIO_MAX = 2.0                            # biggest allowed change of apparent size between matches
DUP_IOU = 0.55                             # a new box this much on top of a tracked person is a duplicate
COLOUR_W = 0.8                             # weight of the kit-colour check


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BoxTracker:
    def __init__(self, fps, max_gap_s=0.9, high_conf=0.25, low_conf=0.12, size=(1.0, 1.0)):
        self.max_age = max(2, int(round(max_gap_s * fps)))
        self.high, self.low = high_conf, low_conf
        self.W, self.H = size
        self.tracks = []          # dicts: id, box, c (cx, cy), v (vx, vy), h, age, hits, col  (stabilised pixels)
        self.next_id = 1

    # --------------------------------------------------------------- geometry
    def _stabilise(self, dets, mat):
        """Boxes in the first frame's picture (camera pan and zoom removed). mat is normalised, 3x3."""
        if mat is None:
            return [tuple(d[:4]) for d in dets]
        out = []
        for d in dets:
            a = mat @ np.array([d[0] / self.W, d[1] / self.H, 1.0])
            b = mat @ np.array([d[2] / self.W, d[3] / self.H, 1.0])
            out.append((float(a[0] * self.W), float(a[1] * self.H), float(b[0] * self.W), float(b[1] * self.H)))
        return out

    # --------------------------------------------------------------- cost
    def _cost(self, tr, box, col):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        h = max(1.0, box[3] - box[1])
        age = tr["age"] + 1
        look = min(age, 2)                                  # never extrapolate far: velocity is a guess
        px, py = tr["c"][0] + tr["v"][0] * look, tr["c"][1] + tr["v"][1] * look
        href = max(tr["h"], h)
        dist = float(np.hypot(px - cx, py - cy)) / href
        gate = GATE_BASE + GATE_PER_FRAME * min(age, 4)     # body-lengths
        ratio = max(tr["h"], h) / max(1.0, min(tr["h"], h))
        if dist > gate or ratio > RATIO_MAX:
            return BIG
        cost = dist + 0.6 * (1.0 - _iou(tr["box"], box))
        if col is not None and tr["col"] is not None:
            cost += COLOUR_W * min(1.0, float(np.linalg.norm(col - tr["col"])) / 45.0)
        return cost

    def _match(self, track_idx, boxes, det_idx, cols):
        if not track_idx or not det_idx:
            return []
        M = np.full((len(track_idx), len(det_idx)), BIG)
        for a, ti in enumerate(track_idx):
            for b, di in enumerate(det_idx):
                M[a, b] = self._cost(self.tracks[ti], boxes[di], cols[di])
        rows, cols_ = linear_sum_assignment(M)
        return [(track_idx[r], det_idx[c]) for r, c in zip(rows, cols_) if M[r, c] < BIG]

    # --------------------------------------------------------------- update
    def update(self, dets, cols=None, mat=None):
        """dets: [(x1, y1, x2, y2, conf), ...] in pixels. cols: optional per-detection colour vectors.
        mat: optional 3x3 (normalised) camera matrix taking this frame to the first one.
        Returns a track id for every detection (None for a weak detection that matched nothing)."""
        cols = cols or [None] * len(dets)
        boxes = self._stabilise(dets, mat)
        ids = [None] * len(dets)
        high = [i for i, d in enumerate(dets) if d[4] >= self.high]
        low = [i for i, d in enumerate(dets) if d[4] < self.high]
        matched_t = set()
        for stage in (high, low):
            pairs = self._match([t for t in range(len(self.tracks)) if t not in matched_t], boxes, stage, cols)
            for ti, di in pairs:
                tr = self.tracks[ti]
                box = boxes[di]
                c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                dt = tr["age"] + 1
                vx = 0.6 * tr["v"][0] + 0.4 * (c[0] - tr["c"][0]) / dt
                vy = 0.6 * tr["v"][1] + 0.4 * (c[1] - tr["c"][1]) / dt
                lim = 0.5 * max(tr["h"], box[3] - box[1])          # nobody moves more than ~0.5 body per frame
                tr.update(box=box, c=c, v=(float(np.clip(vx, -lim, lim)), float(np.clip(vy, -lim, lim))),
                          h=0.7 * tr["h"] + 0.3 * (box[3] - box[1]), age=0, hits=tr["hits"] + 1)
                if cols[di] is not None:
                    tr["col"] = cols[di] if tr["col"] is None else 0.8 * tr["col"] + 0.2 * cols[di]
                ids[di] = tr["id"]
                matched_t.add(ti)
        for ti in range(len(self.tracks)):                          # anyone not seen this frame gets older
            if ti not in matched_t:
                self.tracks[ti]["age"] += 1
        taken = [boxes[di] for di in range(len(dets)) if ids[di] is not None]
        for di in high:                                             # unmatched confident detections start tracks
            if ids[di] is None:
                box = boxes[di]
                # A second box on someone who is already being tracked this frame (a raised racket makes the
                # box jump in size) is the same person, not a new one.
                if any(_iou(box, t) > DUP_IOU for t in taken):
                    continue
                self.tracks.append({"id": self.next_id, "box": box,
                                    "c": ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), "v": (0.0, 0.0),
                                    "h": float(box[3] - box[1]), "age": 0, "hits": 1, "col": cols[di]})
                ids[di] = self.next_id
                self.next_id += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return ids
