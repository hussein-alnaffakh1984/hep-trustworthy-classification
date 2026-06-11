"""
compare_backbones.py - Significance-tested backbone comparison.

Trains a shallow CNN (72 px), EfficientNet-B0 (ImageNet transfer, grayscale replicated
to three channels), and ConvNeXt-Tiny (ImageNet transfer), then reports bootstrap 95%
confidence intervals and McNemar tests against the proposed compact network (whose test
logits are loaded from train.py output). Run train.py first.
"""
import os, glob
import numpy as np
import scipy.io as sio
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from scipy.stats import chi2

SEED, N = 42, 6
tf.random.set_seed(SEED); np.random.seed(SEED)


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
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=SEED)
    p_val, p_te, y_val, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    return p_tr, y_tr, p_val, y_val, p_te, y_te


AUTOTUNE = tf.data.AUTOTUNE
AUG = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal", seed=SEED),
                           tf.keras.layers.RandomRotation(0.12, seed=SEED),
                           tf.keras.layers.RandomZoom(0.1, seed=SEED)])


def run(p_tr, y_tr, p_val, y_val, p_te, y_te, IMG, ch, build, divide=True, lr=1e-3, epochs=25):
    def dec(path, label):
        im = tf.io.read_file(path); im = tf.image.decode_png(im, channels=ch)
        im = tf.image.resize(im, [IMG, IMG]); im = tf.cast(im, tf.float32)
        return (im / 255.0 if divide else im), tf.cast(label, tf.int32)

    def mk(P, Y, tr=False):
        ds = tf.data.Dataset.from_tensor_slices((P, Y))
        if tr:
            ds = ds.shuffle(4096, seed=SEED)
        ds = ds.map(dec, num_parallel_calls=AUTOTUNE).batch(64)
        if tr:
            ds = ds.map(lambda x, z: (AUG(x, training=True), z), num_parallel_calls=AUTOTUNE)
        return ds.prefetch(AUTOTUNE)

    cw = {int(i): float(w) for i, w in
          enumerate(compute_class_weight("balanced", classes=np.arange(N), y=y_tr))}
    m = build(IMG, ch)
    m.compile(optimizer=keras.optimizers.Adam(lr),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    m.fit(mk(p_tr, y_tr, True), validation_data=mk(p_val, y_val), epochs=epochs,
          class_weight=cw, verbose=2,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5,
                                                      restore_best_weights=True),
                     tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    return m.predict(mk(p_te, y_te), verbose=0).argmax(1)


def shallow(IMG, ch):
    inp = tf.keras.Input((IMG, IMG, ch)); x = inp
    for f in [32, 64, 128]:
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(N, activation="softmax")(x))


def transfer(base_fn):
    def build(IMG, ch):
        base = base_fn(include_top=False, weights="imagenet",
                       input_shape=(IMG, IMG, 3), pooling="avg")
        inp = tf.keras.Input((IMG, IMG, 3)); x = base(inp)
        x = tf.keras.layers.Dropout(0.3)(x)
        return tf.keras.Model(inp, tf.keras.layers.Dense(N, activation="softmax")(x))
    return build


def main():
    p_tr, y_tr, p_val, y_val, p_te, y_te = reload_split()
    print(">>> shallow CNN (72 px, grayscale)")
    pred_sh = run(p_tr, y_tr, p_val, y_val, p_te, y_te, 72, 1, shallow, divide=True)
    print(">>> EfficientNet-B0 (96 px, grayscale->3ch, native preprocessing)")
    pred_ef = run(p_tr, y_tr, p_val, y_val, p_te, y_te, 96, 3,
                  transfer(tf.keras.applications.EfficientNetB0), divide=False, lr=1e-3)
    print(">>> ConvNeXt-Tiny (96 px, grayscale->3ch, native preprocessing)")
    pred_cx = run(p_tr, y_tr, p_val, y_val, p_te, y_te, 96, 3,
                  transfer(tf.keras.applications.ConvNeXtTiny), divide=False, lr=1e-4, epochs=20)

    # proposed model predictions from saved logits
    T = float(np.load("temperature.npy")); tl = np.load("test_logits.npy")
    z = tl / T; z -= z.max(1, keepdims=True); e = np.exp(z)
    pred_pr = (e / e.sum(1, keepdims=True)).argmax(1)

    rng = np.random.default_rng(0); nn = len(y_te)
    def ci(pred):
        c = (pred == y_te).astype(float)
        lo, hi = np.percentile([c[rng.integers(0, nn, nn)].mean() for _ in range(1000)], [2.5, 97.5])
        return c.mean(), lo, hi
    print("\n=== Architecture comparison (accuracy [95% CI], macro-F1) ===")
    for nm, pr in [("Shallow 72px", pred_sh), ("EfficientNet-B0", pred_ef),
                   ("ConvNeXt-Tiny", pred_cx), ("Proposed", pred_pr)]:
        a, lo, hi = ci(pr)
        print(f"  {nm:18s} {a:.4f} [{lo:.4f}, {hi:.4f}]  F1={f1_score(y_te, pr, average='macro'):.4f}")
    print("\n=== McNemar (proposed vs. each) ===")
    for nm, pr in [("Shallow", pred_sh), ("EfficientNet", pred_ef), ("ConvNeXt", pred_cx)]:
        pc = (pred_pr == y_te); oc = (pr == y_te)
        n10 = int((pc & ~oc).sum()); n01 = int((~pc & oc).sum())
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if (n10 + n01) > 0 else 0.0
        print(f"  vs {nm:12s}: prop_only={n10} other_only={n01} chi2={stat:.1f} p={1 - chi2.cdf(stat, 1):.2e}")


if __name__ == "__main__":
    main()
