"""
mahalanobis_ood.py - Feature-space Mahalanobis out-of-distribution detector.

A class-conditional Gaussian with a shared (tied) covariance is fit on the training
features. The OOD score of a sample is its minimum Mahalanobis distance to the class
centroids; clean vs. corrupted inputs are then separated by AUROC. This recovers the
detection that the maximum-softmax-probability and energy scores miss under noise.
Run train.py first.
"""
import os, glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

IMG, N = 96, 6


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
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=42)
    _, p_te, _, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)
    return p_tr, y_tr, p_te, y_te


def main():
    p_tr, y_tr, p_te, y_te = reload_split()
    model = tf.keras.models.load_model("ana_final_model.keras")
    feat = tf.keras.Model(model.input, model.layers[-1].input)

    def load(path):
        im = tf.io.read_file(path); im = tf.image.decode_png(im, channels=1)
        im = tf.image.resize(im, [IMG, IMG]); return tf.cast(im, tf.float32) / 255.0

    def feats(P, kind=None, sev=0.0):
        def f(path):
            x = load(path)
            if kind == "noise":
                x = x + tf.random.stateless_normal(tf.shape(x), [7, int(sev * 1000) + 1], 0.0, sev)
            elif kind == "brightness":
                x = x * (1.0 - sev)
            elif kind == "contrast":
                x = tf.image.adjust_contrast(x, 1.0 - sev)
            return tf.clip_by_value(x, 0.0, 1.0)
        return feat.predict(tf.data.Dataset.from_tensor_slices(P).map(f).batch(128), verbose=0)

    print("fitting class-conditional Gaussian on training features ...")
    Ftr = feats(p_tr); D = Ftr.shape[1]
    mus = np.stack([Ftr[y_tr == c].mean(0) for c in range(N)])
    Xc = Ftr - mus[y_tr]
    cov_inv = np.linalg.inv(np.cov(Xc.T) + 1e-3 * np.eye(D))

    def maha(F):  # min Mahalanobis distance to a class centroid; higher = more OOD
        d = np.empty((len(F), N))
        for c in range(N):
            diff = F - mus[c]; d[:, c] = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        return d.min(1)

    s_clean = maha(feats(p_te))
    print("\n=== Mahalanobis OOD detection AUROC (clean vs. corrupted) ===")
    for kind, sev in {"noise": 0.15, "brightness": 0.40, "contrast": 0.60}.items():
        sc = maha(feats(p_te, kind, sev))
        yy = np.r_[np.zeros(len(s_clean)), np.ones(len(sc))]
        print(f"  {kind:10s}: Mahalanobis-AUROC={roc_auc_score(yy, np.r_[s_clean, sc]):.3f}")


if __name__ == "__main__":
    main()
