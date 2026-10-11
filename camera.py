"""Camera motion for hand-held and panning footage.

A phone filming from the sideline pans and zooms. If that is ignored, everyone standing still appears to run
several body-lengths, so the crowd, the bench and the staff look like players. This estimates the pan AND zoom
between analysed frames by following background features (optical flow) and fitting a similarity transform with
RANSAC (moving players are just outliers), then chains the steps back to the first frame.

M is a 3x3 matrix in *normalised* picture coordinates (0..1) that maps a point in the current frame to where the
same ground point was in the first frame.
"""
import cv2
import numpy as np


class CameraTracker:
    WIDTH = 480

    def __init__(self, W, H):
        self.cw = self.WIDTH
        self.ch = max(8, int(round(self.WIDTH * H / W)))
        self.S = np.diag([self.cw, self.ch, 1.0])
        self.Sinv = np.linalg.inv(self.S)
        self.window = cv2.createHanningWindow((self.cw, self.ch), cv2.CV_32F)
        self.prev = None
        self.M = np.eye(3)
        self.steps = 0
        self.good_steps = 0

    def _estimate(self, prev, cur):
        """3x3 (gray-pixel coords) taking points of `cur` to `prev`, or None when it cannot be trusted."""
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.008, minDistance=6, blockSize=7)
        if p0 is None or len(p0) < 20:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        good = st.ravel() == 1
        if good.sum() < 20:
            return None
        A, inl = cv2.estimateAffinePartial2D(p1[good], p0[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=1.5, maxIters=2000, confidence=0.99)
        if A is None or inl is None or int(inl.sum()) < 15 or inl.sum() < 0.3 * good.sum():
            return None
        scale = float(np.hypot(A[0, 0], A[1, 0]))
        if not 0.8 <= scale <= 1.25 or abs(A[0, 2]) > 0.35 * self.cw or abs(A[1, 2]) > 0.35 * self.ch:
            return None
        return np.vstack([A, [0.0, 0.0, 1.0]])

    def update(self, frame):
        """Feed the next analysed frame; returns the chained matrix (normalised coordinates)."""
        g = cv2.cvtColor(cv2.resize(frame, (self.cw, self.ch), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            self.steps += 1
            A = self._estimate(self.prev, g)
            if A is None:                                   # featureless (plain court, ice): fall back to a shift
                (sx, sy), strength = cv2.phaseCorrelate(self.prev.astype(np.float32), g.astype(np.float32),
                                                        self.window)
                if strength > 0.05:
                    A = np.array([[1.0, 0.0, -sx], [0.0, 1.0, -sy], [0.0, 0.0, 1.0]])
            else:
                self.good_steps += 1
            if A is not None:
                self.M = self.M @ (self.Sinv @ A @ self.S)
        self.prev = g
        return self.M

    def centre_shift(self):
        """How far the middle of the picture has drifted from the first frame (old-style 'cam')."""
        c = self.M @ np.array([0.5, 0.5, 1.0])
        return float(0.5 - c[0]), float(0.5 - c[1])
