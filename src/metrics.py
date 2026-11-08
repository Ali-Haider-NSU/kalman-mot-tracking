"""CLEAR-MOT (MOTA, ID switches) and IDF1."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracker import iou_matrix


def evaluate(gt, results, iou_min=0.5):
    """gt and results: per-frame lists of (id, box). Returns MOTA, IDF1, IDSW, FP, FN."""
    fn = fp = idsw = n_gt = n_trk = 0
    last = {}                  # gt id -> track id it was last matched to
    pair_count = {}            # (gt id, track id) -> frames matched (for IDF1)
    for g, r in zip(gt, results):
        n_gt += len(g)
        n_trk += len(r)
        pairs = []
        if len(g) and len(r):
            iou = iou_matrix(np.array([b for _, b in g]), np.array([b for _, b in r]))
            rows, cols = linear_sum_assignment(-iou)
            pairs = [(i, j) for i, j in zip(rows, cols) if iou[i, j] >= iou_min]
        for i, j in pairs:
            gid, tid = g[i][0], r[j][0]
            if gid in last and last[gid] != tid:
                idsw += 1      # same object, new track id: identity switch
            last[gid] = tid
            pair_count[(gid, tid)] = pair_count.get((gid, tid), 0) + 1
        fn += len(g) - len(pairs)
        fp += len(r) - len(pairs)
    mota = 1 - (fn + fp + idsw) / n_gt if n_gt else 0.0
    idf1 = 0.0
    if pair_count:             # one-to-one assignment of gt ids to track ids over the sequence
        gids = sorted({k[0] for k in pair_count})
        tids = sorted({k[1] for k in pair_count})
        C = np.zeros((len(gids), len(tids)))
        for (gid, tid), c in pair_count.items():
            C[gids.index(gid), tids.index(tid)] = c
        rows, cols = linear_sum_assignment(-C)
        idf1 = 2 * C[rows, cols].sum() / (n_gt + n_trk)
    return dict(MOTA=mota, IDF1=idf1, IDSW=idsw, FP=fp, FN=fn)
