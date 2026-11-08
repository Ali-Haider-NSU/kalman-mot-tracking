"""Unit tests: run with `pytest` from the repository root."""
import numpy as np
import pytest

from src.kalman_filter import KalmanFilter, constant_velocity_model, kf1d_step
from src.tracker import SORT, associate, iou_matrix
from src.metrics import evaluate


def test_kf1d_converges_and_shrinks_variance():
    x, P = 0.0, 100.0
    for z in [10.2, 9.7, 10.4]:
        x, P, K = kf1d_step(x, P, z, q=0.01, r=0.25)
    assert abs(x - 10.0) < 0.3
    assert P < 0.25                      # more certain than a single measurement


def test_kf1d_gain_limits():
    # huge prior uncertainty: trust the measurement; tiny uncertainty: ignore it
    _, _, K = kf1d_step(0.0, 1e6, 5.0, q=0.0, r=1.0)
    assert K == pytest.approx(1.0, abs=1e-5)
    _, _, K = kf1d_step(0.0, 1e-6, 5.0, q=0.0, r=1.0)
    assert K == pytest.approx(0.0, abs=1e-5)


def test_constant_velocity_learns_velocity():
    dt = 0.1
    F, H, Q, R = constant_velocity_model(dt, q=0.5, r=0.1)
    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(4), P0=np.diag([1, 1, 100, 100]))
    rng = np.random.default_rng(0)
    for k in range(1, 100):                      # object moving at (2, -1) m/s
        kf.predict()
        kf.update(np.array([2.0, -1.0]) * k * dt + rng.normal(0, 0.1, 2))
    assert kf.x[2:] == pytest.approx([2.0, -1.0], abs=0.2)


def test_covariance_stays_symmetric_positive_definite():
    F, H, Q, R = constant_velocity_model(0.05, q=1.0, r=0.5)
    kf = KalmanFilter(F, H, Q, R, x0=np.zeros(4), P0=np.eye(4))
    for _ in range(50):
        kf.predict()
        kf.update(np.zeros(2))
    assert np.allclose(kf.P, kf.P.T, atol=1e-9)
    assert np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_iou_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 10, 10]])
    assert iou_matrix(a, b)[0] == pytest.approx([50 / 150, 0.0, 1.0])


def test_associate_with_gating():
    tracks = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    dets = np.array([[1, 0, 11, 10], [300, 300, 310, 310]])
    matches, um_t, um_d = associate(tracks, dets, iou_min=0.3)
    assert matches == [(0, 0)]
    assert um_t == [1] and um_d == [1]


def test_sort_keeps_id_of_moving_box():
    trk = SORT(min_hits=1)
    ids = set()
    for f in range(20):                          # one box moving 5 px per frame
        det = np.array([[100 + 5 * f, 100, 150 + 5 * f, 200, 0.9]])
        out = trk.step(det)
        ids |= {tid for tid, _ in out}
    assert ids == {1}


def test_perfect_tracking_scores_one():
    box = np.array([0, 0, 10, 10.0])
    gt = [[(0, box)] for _ in range(5)]
    res = [[(7, box)] for _ in range(5)]
    m = evaluate(gt, res)
    assert m["MOTA"] == 1.0 and m["IDF1"] == 1.0 and m["IDSW"] == 0


def test_identity_switch_is_counted():
    box = np.array([0, 0, 10, 10.0])
    gt = [[(0, box)] for _ in range(4)]
    res = [[(1, box)], [(1, box)], [(2, box)], [(2, box)]]
    assert evaluate(gt, res)["IDSW"] == 1
