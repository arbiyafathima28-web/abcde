import unittest
import numpy as np
from app.dataset import DatasetStore

class TestDataset(unittest.TestCase):
    def test_ensure_demo(self):
        store = DatasetStore()
        store.ensure_demo()
        summary = store.summary()
        self.assertEqual(summary["name"], "Synthetic demo data")
        self.assertTrue(summary["synthetic"])
        self.assertEqual(summary["train_samples"], 900)
        self.assertEqual(summary["val_samples"], 150)
        self.assertEqual(summary["test_samples"], 150)
        self.assertEqual(summary["features"], 32)

if __name__ == "__main__":
    unittest.main()
