"""
robustness_ood.py - Corruption robustness and out-of-distribution detection.

Loads the trained model + saved test split. Measures accuracy and mean confidence
under Gaussian noise, reduced brightness, and reduced contrast (fixed seed), then
reports OOD-detection AUROC (clean vs. corrupted) for the maximum-softmax-probability
and energy scores. Run train.py first.
"""
import os, glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

IMG = 96


def find_data():
    hits = glob.glob("/kaggle/input/**/labels.mat", recursive=True) or \
           glob.glob("./data/**/labels.mat", recursive=True)
    base = os.path.dirname(hits[0]); return base, os.path.join(base, "cells", "cells")


def reload_split():
    base, img_dir = find_data()
    labels = sio.loadmat(os.path.join(base, "labels.mat"))["labels"].flatten()
    present = set(os.listdir(img_dir)); paths, y = [], []
    for n in range(1, len(labels) + 1):
        if f"{n}.png" in present:
            paths.append(os.path.join(img_dir, f"{n}.png")); y.append(int(labels[n - 1]) - 1)
    paths, y = np.array(paths), np.array(y, dtype=np.int32)
    _, p_tmp, _, y_tmp = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=42)
    _, p_te, _, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    return p_te, y_te


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main():
    T = float(np.load("temperature.npy"))
    p_te = np.load("p_te.npy"); y_te = np.load("y_te.npy")
    model = tf.keras.models.load_model("ana_final_model.keras")
    last = model.layers[-1]; W, b = last.get_weights(); feat = tf.keras.Model(model.input, last.input)

    def load(path):
        im = tf.io.read_file(path); im = tf.image.decode_png(im, channels=1)
        im = tf.image.resize(im, [96, 96]); return tf.cast(im, tf.float32) / 255.0

    def logits(kind, sev):
        def f(path):
            x = load(path)
            if kind == "noise":
                x = x + tf.random.stateless_normal(tf.shape(x), [7, int(sev * 1000) + 1], 0.0, sev)
            elif kind == "brightness":
                x = x * (1.0 - sev)
            elif kind == "contrast":
                x = tf.image.adjust_contrast(x, 1.0 - sev)
            return tf.clip_by_value(x, 0.0, 1.0)
        ds = tf.data.Dataset.from_tensor_slices(p_te).map(f).batch(64)
        return feat.predict(ds, verbose=0) @ W + b

    grid = {"noise": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
            "brightness": [0.0, 0.20, 0.40, 0.60],
            "contrast": [0.0, 0.30, 0.60, 0.90]}
    clean = logits("noise", 0.0)
    print("=== Robustness (fixed-seed corruption) ===")
    for kind in grid:
        for sev in grid[kind]:
            lg = clean if sev == 0.0 else logits(kind, sev)
            p = softmax(lg / T); pred = p.argmax(1); acc = (pred == y_te).mean()
            top = Counter(pred).most_common(1)[0]
            print(f"  {kind:10s} sev={sev:.2f} acc={acc:.3f} conf={p.max(1).mean():.3f} "
                  f"top=class{top[0]}({top[1]*100//len(pred)}%)")

    def msp(lg):
        return 1.0 - softmax(lg).max(1)
    def energy(lg):
        m = lg.max(1, keepdims=True); return -(m.squeeze(1) + np.log(np.exp(lg - m).sum(1)))
    print("\n=== OOD detection AUROC (clean vs. corrupted) ===")
    for kind, sev in {"noise": 0.15, "brightness": 0.40, "contrast": 0.60}.items():
        c = logits(kind, sev); yy = np.r_[np.zeros(len(clean)), np.ones(len(c))]
        print(f"  {kind:10s}: MSP={roc_auc_score(yy, np.r_[msp(clean), msp(c)]):.3f} "
              f"Energy={roc_auc_score(yy, np.r_[energy(clean), energy(c)]):.3f}")


if __name__ == "__main__":
    main()
