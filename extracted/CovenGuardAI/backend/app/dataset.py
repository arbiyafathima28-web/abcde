from __future__ import annotations
import io
import numpy as np

class DatasetStore:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        self.name = "Synthetic demo data"
        self.synthetic = True

    def load_npz(self, raw: bytes):
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        required = ["train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"]
        missing = [k for k in required if k not in z.files]
        if missing:
            raise ValueError("Missing arrays in NPZ: " + ", ".join(missing))
        self.train_x, self.train_y = self._prepare(z["train_images"], z["train_labels"])
        self.val_x, self.val_y = self._prepare(z["val_images"], z["val_labels"])
        self.test_x, self.test_y = self._prepare(z["test_images"], z["test_labels"])
        self.name = "PneumoniaMNIST"
        self.synthetic = False
        return self.summary()

    @staticmethod
    def _prepare(images, labels):
        x = np.asarray(images, dtype=np.float32)
        if x.ndim == 3:
            x = x[..., None]
        if x.ndim != 4:
            raise ValueError(f"Expected image array with 3 or 4 dimensions; got {x.shape}")
        if x.shape[-1] not in (1, 3) and x.shape[1] in (1, 3):
            x = np.transpose(x, (0, 2, 3, 1))
        x = x.reshape(len(x), -1)
        x /= 255.0 if x.max(initial=0) > 1 else 1.0
        y = np.asarray(labels).reshape(-1).astype(np.int64)
        if len(x) != len(y):
            raise ValueError("Image and label counts do not match")
        # MedMNIST PneumoniaMNIST labels are 0/1.
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("This MVP expects binary labels 0 and 1.")
        return x, y

    def ensure_demo(self):
        if self.train_x is not None:
            return
        rng = np.random.default_rng(42)
        # Explicitly synthetic feature data for a functional baseline.
        n, d = 1200, 32
        x = rng.normal(0, 1, (n, d)).astype(np.float32)
        score = x[:, 0] * 1.1 + x[:, 1] * 0.8 - x[:, 2] * 0.5 + rng.normal(0, .6, n)
        y = (score > np.median(score)).astype(np.int64)
        self.train_x, self.train_y = x[:900], y[:900]
        self.val_x, self.val_y = x[900:1050], y[900:1050]
        self.test_x, self.test_y = x[1050:], y[1050:]
        self.name = "Synthetic demo data"
        self.synthetic = True

    def summary(self):
        return {
            "name": self.name, "synthetic": self.synthetic,
            "train_samples": int(len(self.train_y)), "val_samples": int(len(self.val_y)),
            "test_samples": int(len(self.test_y)),
            "features": int(self.train_x.shape[1]),
            "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(self.train_y, return_counts=True))}
        }
