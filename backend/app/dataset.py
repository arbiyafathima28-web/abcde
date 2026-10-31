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
            # (N, H, W) -> (N, 1, H, W)
            x = x[:, None, :, :]
        elif x.ndim == 4:
            # Check channel dimension: if (N, H, W, 1), convert to (N, 1, H, W)
            if x.shape[-1] == 1:
                x = np.transpose(x, (0, 3, 1, 2))
            elif x.shape[1] != 1 and x.shape[-1] in (1, 3):
                x = np.transpose(x, (0, 3, 1, 2))
                # If RGB, convert to single channel
                if x.shape[1] == 3:
                    x = x[:, 0:1, :, :]
        elif x.ndim == 2 and x.shape[1] == 784:
            x = x.reshape(-1, 1, 28, 28)
        else:
            raise ValueError(f"Expected image array with 3 or 4 dimensions; got {x.shape}")

        if x.shape[2:] != (28, 28):
            raise ValueError(f"Expected 28x28 image spatial resolution; got {x.shape[2:]}")

        # Normalization: map [0, 255] or [0, 1] to [-1.0, 1.0] standard zero-mean
        if x.max(initial=0) > 1.0:
            x = x / 255.0
        x = (x - 0.5) / 0.5

        y = np.asarray(labels).reshape(-1).astype(np.int64)
        if len(x) != len(y):
            raise ValueError("Image and label counts do not match")
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("PneumoniaMNIST expects binary labels 0 (Normal) and 1 (Pneumonia).")
        return x, y

    def ensure_demo(self):
        if self.train_x is not None:
            return
        rng = np.random.default_rng(42)
        n = 1200
        images = np.zeros((n, 1, 28, 28), dtype=np.float32)
        labels = rng.integers(0, 2, n, dtype=np.int64)
        yy, xx = np.mgrid[0:28, 0:28]

        for i in range(n):
            # Lung contours
            left_lobe = ((xx - 9)**2 / 16.0 + (yy - 14)**2 / 36.0) < 1.0
            right_lobe = ((xx - 19)**2 / 16.0 + (yy - 14)**2 / 36.0) < 1.0
            lungs = (left_lobe | right_lobe).astype(np.float32) * 0.45
            noise = rng.normal(0, 0.07, (28, 28)).astype(np.float32)
            img = lungs + noise
            # Pneumonia consolidation opacity
            if labels[i] == 1:
                cx, cy = rng.choice([9, 19]), rng.integers(12, 20)
                img += np.exp(-((xx - cx)**2 + (yy - cy)**2) / 12.0) * 0.75
            img = np.clip(img, 0.0, 1.0)
            images[i, 0] = (img - 0.5) / 0.5

        self.train_x, self.train_y = images[:900], labels[:900]
        self.val_x, self.val_y = images[900:1050], labels[900:1050]
        self.test_x, self.test_y = images[1050:], labels[1050:]
        self.name = "Synthetic PneumoniaMNIST"
        self.synthetic = True

    def summary(self):
        features = int(self.train_x.shape[2] * self.train_x.shape[3]) if self.train_x.ndim == 4 else int(self.train_x.shape[1])
        return {
            "name": self.name,
            "synthetic": self.synthetic,
            "train_samples": int(len(self.train_y)),
            "val_samples": int(len(self.val_y)),
            "test_samples": int(len(self.test_y)),
            "features": features,
            "class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(self.train_y, return_counts=True))}
        }
