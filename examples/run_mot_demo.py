"""Compare tracker variants on the synthetic scenario and print a Markdown table for the README.

Run from the repository root:  python -m examples.run_mot_demo
"""
import numpy as np

from src.simulation import make_objects, simulate
from src.tracker import SORT, ByteTrackLike, run
from src.metrics import evaluate


def main(seed=7):
    rng = np.random.default_rng(seed)
    objects = make_objects(rng)
    gt, dets = simulate(objects, rng)
    print(f"{len(objects)} objects, {sum(len(g) for g in gt)} ground-truth boxes, "
          f"{sum(len(d) for d in dets)} detections in {len(gt)} frames\n")

    trackers = [("IoU only, no Kalman", SORT(use_kf=False)),
                ("SORT, max_age = 1", SORT(max_age=1)),
                ("SORT, min_hits = 1", SORT(min_hits=1)),
                ("SORT", SORT()),
                ("ByteTrack-like", ByteTrackLike())]
    print("| Tracker | MOTA | IDF1 | ID switches | FP | FN |")
    print("|---|---|---|---|---|---|")
    for name, trk in trackers:
        m = evaluate(gt, run(trk, dets))
        print(f"| {name} | {m['MOTA']:.3f} | {m['IDF1']:.3f} | {m['IDSW']} | {m['FP']} | {m['FN']} |")


if __name__ == "__main__":
    main()
