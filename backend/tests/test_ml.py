import unittest
import numpy as np
from app.ml import sigmoid, predict, train_local, metrics, fedavg, coordinate_median, trimmed_mean, inspect_updates

class TestML(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertTrue(0.0 <= sigmoid(-100) <= 0.01)
        self.assertTrue(0.99 <= sigmoid(100) <= 1.0)

    def test_fedavg(self):
        u1 = np.array([1.0, 2.0, 3.0])
        u2 = np.array([3.0, 4.0, 5.0])
        avg = fedavg([u1, u2], [1, 1])
        np.testing.assert_allclose(avg, [2.0, 3.0, 4.0])

    def test_coordinate_median(self):
        u1 = np.array([1.0, 10.0])
        u2 = np.array([2.0, 20.0])
        u3 = np.array([100.0, 30.0])  # u3 has an extreme outlier in first dimension
        med = coordinate_median([u1, u2, u3])
        np.testing.assert_allclose(med, [2.0, 20.0])

    def test_trimmed_mean(self):
        updates = [np.array([float(i)]) for i in range(10)]
        # trimmed mean should remove extremes
        res = trimmed_mean(updates, trim_ratio=0.2)
        self.assertAlmostEqual(float(res[0]), 4.5)

    def test_anomaly_detection_sign_flip(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0.5, 0.1, size=(9, 10))
        # 10th update is sign-flipped / scaled adversary
        poisoned = -4.0 * np.mean(base, axis=0)
        updates = [base[i] for i in range(9)] + [poisoned]
        norms, sim, dev, robust_z, flags, cosine = inspect_updates(updates)
        # Client 10 (index 9) must be flagged
        self.assertTrue(flags[9], "Poisoned client was not flagged by anomaly detector")

if __name__ == "__main__":
    unittest.main()
