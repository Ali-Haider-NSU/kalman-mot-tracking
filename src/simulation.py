"""Synthetic MOT scenario with ground truth: pedestrians crossing a 1280x720 image, a pillar
that occludes them, detector noise, missed detections and false positives."""
import numpy as np

W_IMG, H_IMG = 1280, 720
PILLAR = (600, 700)          # x-range of an occluding pillar


def make_objects(rng, n_obj=12):
    objs = []
    for i in range(n_obj):
        start = rng.integers(0, 200)
        life = rng.integers(120, 260)
        left = rng.random() < 0.5
        w, h = rng.uniform(40, 90), rng.uniform(90, 180)
        x0 = -w / 2 if left else W_IMG + w / 2
        vx = rng.uniform(4, 8) * (1 if left else -1)          # px per frame
        objs.append(dict(id=i, start=start, end=start + life, x=x0, y=rng.uniform(150, 560),
                         vx=vx, vy=rng.uniform(-0.6, 0.6), w=w, h=h))
    return objs


def simulate(objs, rng, n_frames=300):
    """Return per-frame ground truth [(id, box)] and detections (N x 5: x1 y1 x2 y2 score)."""
    gt, dets = [], []
    state = {o["id"]: dict(o) for o in objs}
    for f in range(n_frames):
        g, d = [], []
        for o in state.values():
            if not (o["start"] <= f < o["end"]):
                continue
            o["vx"] += rng.normal(0, 0.05)
            o["vy"] += rng.normal(0, 0.05)
            o["x"] += o["vx"]
            o["y"] += o["vy"]
            box = np.array([o["x"] - o["w"] / 2, o["y"] - o["h"] / 2,
                            o["x"] + o["w"] / 2, o["y"] + o["h"] / 2])
            if box[2] < 0 or box[0] > W_IMG:
                continue
            g.append((o["id"], box))
            overlap = max(0, min(box[2], PILLAR[1]) - max(box[0], PILLAR[0]))
            hidden = overlap / (box[2] - box[0])
            if hidden > 0.7:                                   # fully occluded: no detection
                continue
            score = rng.uniform(0.6, 0.95) if hidden < 0.3 else rng.uniform(0.2, 0.45)
            if rng.random() < 0.08:                            # random miss
                continue
            c = (box[:2] + box[2:]) / 2 + rng.normal(0, 3, 2)
            wh = (box[2:] - box[:2]) * (1 + rng.normal(0, 0.04, 2))
            d.append(np.r_[c - wh / 2, c + wh / 2, score])
        for _ in range(rng.poisson(1.5)):                      # false positives (clutter)
            c = rng.uniform([0, 100], [W_IMG, 650])
            wh = rng.uniform([30, 60], [90, 160])
            score = rng.uniform(0.5, 0.75) if rng.random() < 0.2 else rng.uniform(0.1, 0.5)
            d.append(np.r_[c - wh / 2, c + wh / 2, score])
        gt.append(g)
        dets.append(np.array(d).reshape(-1, 5))
    return gt, dets
