"""SORT-style multi-object tracker: Kalman prediction, IoU + Hungarian matching, track lifecycle."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter


def iou_matrix(a, b):
    """IoU between every box in a (N x 4) and every box in b (M x 4), format x1 y1 x2 y2."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def associate(track_boxes, det_boxes, iou_min=0.3):
    """Hungarian matching on 1 - IoU; pairs below iou_min are rejected (gating).

    Returns (matches, unmatched track indices, unmatched detection indices).
    """
    iou = iou_matrix(track_boxes, det_boxes)
    if iou.size == 0:
        return [], list(range(len(iou))), list(range(iou.shape[1]))
    rows, cols = linear_sum_assignment(1 - iou)
    matches = [(r, c) for r, c in zip(rows, cols) if iou[r, c] >= iou_min]
    mt = {r for r, _ in matches}
    md = {c for _, c in matches}
    return (matches, [r for r in range(iou.shape[0]) if r not in mt],
            [c for c in range(iou.shape[1]) if c not in md])


def box_to_z(b):
    """x1 y1 x2 y2 -> cx cy w h"""
    return np.r_[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]]


def x_to_box(x):
    """state [cx, cy, w, h, ...] -> x1 y1 x2 y2"""
    cx, cy, w, h = x[:4]
    return np.r_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class Track:
    """One object: a Kalman filter on [cx, cy, w, h, vcx, vcy, vw, vh] plus bookkeeping."""
    _next_id = 1

    def __init__(self, box):
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)                                   # dt = 1 frame
        H = np.eye(4, 8)
        Q = np.diag([1, 1, 1, 1, 0.5, 0.5, 0.1, 0.1]) ** 2
        R = np.diag([3, 3, 4, 6]) ** 2
        P0 = np.diag([3, 3, 4, 6, 20, 20, 5, 5]) ** 2           # velocity unknown at birth
        self.kf = KalmanFilter(F, H, Q, R, np.r_[box_to_z(box), np.zeros(4)], P0)
        self.id = Track._next_id
        Track._next_id += 1
        self.hits, self.age_since_update = 1, 0

    def predict(self):
        self.kf.predict()
        self.age_since_update += 1
        return x_to_box(self.kf.x)

    def update(self, box):
        self.kf.update(box_to_z(box))
        self.hits += 1
        self.age_since_update = 0


class SORT:
    """Simple Online and Realtime Tracking.

    det per frame: array (N x 5) with x1 y1 x2 y2 score. step() returns [(track id, box), ...].
    """

    def __init__(self, iou_min=0.3, max_age=30, min_hits=3, score_min=0.5, use_kf=True):
        self.tracks, self.iou_min, self.max_age = [], iou_min, max_age
        self.min_hits, self.score_min, self.use_kf = min_hits, score_min, use_kf

    def step(self, det):
        det = np.asarray(det, float).reshape(-1, 5)
        det = det[det[:, 4] >= self.score_min]
        if self.use_kf:                                          # 1. predict
            pred = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        else:                                                    # no motion model: last box
            for t in self.tracks:
                t.age_since_update += 1
            pred = np.array([x_to_box(t.kf.x) for t in self.tracks]).reshape(-1, 4)
        matches, _, um_d = associate(pred, det[:, :4], self.iou_min)   # 2. match
        for ti, di in matches:                                   # 3. update
            if self.use_kf:
                self.tracks[ti].update(det[di, :4])
            else:
                t = self.tracks[ti]
                t.kf.x[:4] = box_to_z(det[di, :4])
                t.hits += 1
                t.age_since_update = 0
        for di in um_d:                                          # 4. birth
            self.tracks.append(Track(det[di, :4]))
        self.tracks = [t for t in self.tracks if t.age_since_update <= self.max_age]   # 5. death
        return [(t.id, x_to_box(t.kf.x)) for t in self.tracks
                if t.age_since_update == 0 and t.hits >= self.min_hits]


class ByteTrackLike(SORT):
    """SORT + ByteTrack's idea: low-score detections may update tracks left unmatched."""

    def __init__(self, high=0.5, low=0.1, iou_low=0.5, **kw):
        super().__init__(score_min=high, **kw)
        self.high, self.low, self.iou_low = high, low, iou_low

    def step(self, det):
        det = np.asarray(det, float).reshape(-1, 5)
        hi = det[det[:, 4] >= self.high]
        lo = det[(det[:, 4] >= self.low) & (det[:, 4] < self.high)]
        pred = np.array([t.predict() for t in self.tracks]).reshape(-1, 4)
        m1, um_t, um_d = associate(pred, hi[:, :4], self.iou_min)      # round 1: high scores
        for ti, di in m1:
            self.tracks[ti].update(hi[di, :4])
        m2, _, _ = associate(pred[um_t], lo[:, :4], self.iou_low)      # round 2: low scores
        for k, di in m2:
            self.tracks[um_t[k]].update(lo[di, :4])
        for di in um_d:                                                # births only from high
            self.tracks.append(Track(hi[di, :4]))
        self.tracks = [t for t in self.tracks if t.age_since_update <= self.max_age]
        return [(t.id, x_to_box(t.kf.x)) for t in self.tracks
                if t.age_since_update == 0 and t.hits >= self.min_hits]


def run(tracker, detections):
    """Run a tracker over a sequence; track ids restart at 1."""
    Track._next_id = 1
    return [tracker.step(d) for d in detections]
