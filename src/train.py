"""
train.py - Train the compact CNN and compute the core trustworthy-AI metrics.

Outputs (saved to the working directory, reused by the analysis scripts):
    ana_final_model.keras, temperature.npy,
    val_logits.npy, test_logits.npy, y_val.npy, y_te.npy, p_val.npy, p_te.npy

Prints: classification report, numerical confusion matrix, calibration (4 dp),
        bootstrap 95% CIs, selective prediction / clinical utility, per-pattern
        reliability, and split-conformal prediction (independent calibration set).

Reproducible: seed = 42.
"""
import os, glob, random
import numpy as np
import scipy.io as sio
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.optimize import minimize_scalar

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

CLASS = ["Homogeneous", "Speckled", "Nucleolar", "Centromere", "NuclearMembrane", "Golgi"]
N, IMG = 6, 96


def find_data():
    hits = glob.glob("/kaggle/input/**/labels.mat", recursive=True) or \
           glob.glob("./data/**/labels.mat", recursive=True)
    if not hits:
        raise FileNotFoundError("labels.mat not found. Put the dataset under ./data/ or /kaggle/input/")
    base = os.path.dirname(hits[0])
    return base, os.path.join(base, "cells", "cells")


def load_paths_labels():
    base, img_dir = find_data()
    labels = sio.loadmat(os.path.join(base, "labels.mat"))["labels"].flatten()
    present = set(os.listdir(img_dir))
    paths, y = [], []
    for n in range(1, len(labels) + 1):
        if f"{n}.png" in present:
            paths.append(os.path.join(img_dir, f"{n}.png"))
            y.append(int(labels[n - 1]) - 1)
    return np.array(paths), np.array(y, dtype=np.int32)


def decode(path, label):
    im = tf.io.read_file(path)
    im = tf.image.decode_png(im, channels=1)
    im = tf.image.resize(im, [IMG, IMG])
    return tf.cast(im, tf.float32) / 255.0, tf.cast(label, tf.int32)


AUTOTUNE = tf.data.AUTOTUNE
AUG = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.12, seed=SEED),
    tf.keras.layers.RandomZoom(0.1, seed=SEED),
    tf.keras.layers.RandomContrast(0.1, seed=SEED),
], name="augmentation")


def make_ds(P, Y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((P, Y))
    if training:
        ds = ds.shuffle(4096, seed=SEED)
    ds = ds.map(decode, num_parallel_calls=AUTOTUNE).batch(64)
    if training:  # augmentation applied in the data pipeline (Keras 3 friendly)
        ds = ds.map(lambda x, z: (AUG(x, training=True), z), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def build_model():
    def block(x, f):
        for _ in range(2):
            x = tf.keras.layers.Conv2D(f, 3, padding="same")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation("relu")(x)
        return tf.keras.layers.MaxPooling2D()(x)
    inp = tf.keras.Input((IMG, IMG, 1)); x = inp
    for f in [32, 64, 128, 256]:
        x = block(x, f)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(N, activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    # NOTE: use explicit Keras objects, not string shortcuts (Keras 3 dtype issue).
    m.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return m


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(P, y, nb=15):
    conf = P.max(1); corr = (P.argmax(1) == y).astype(float)
    b = np.linspace(0, 1, nb + 1); o = 0.0
    for i in range(nb):
        m = (conf > b[i]) & (conf <= b[i + 1])
        if m.sum() > 0:
            o += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return o


def brier(P, y):
    return np.mean(np.sum((P - np.eye(N)[y]) ** 2, 1))


def nll(P, y):
    return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))


def main():
    paths, y = load_paths_labels()
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(paths, y, test_size=0.30, stratify=y, random_state=SEED)
    p_val, p_te, y_val, y_te = train_test_split(p_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print("split:", len(y_tr), len(y_val), len(y_te))

    model = build_model()
    cw = {int(i): float(w) for i, w in
          enumerate(compute_class_weight("balanced", classes=np.arange(N), y=y_tr))}
    model.fit(make_ds(p_tr, y_tr, True), validation_data=make_ds(p_val, y_val),
              epochs=40, class_weight=cw, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                                          restore_best_weights=True),
                         tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)])
    model.save("ana_final_model.keras")

    last = model.layers[-1]; W, b = last.get_weights()
    feat = tf.keras.Model(model.input, last.input)
    val_lg = feat.predict(make_ds(p_val, y_val), verbose=0) @ W + b
    test_lg = feat.predict(make_ds(p_te, y_te), verbose=0) @ W + b
    for nm, arr in [("val_logits", val_lg), ("test_logits", test_lg),
                    ("y_val", y_val), ("y_te", y_te), ("p_te", p_te), ("p_val", p_val)]:
        np.save(f"{nm}.npy", arr)

    # temperature scaling (fit on full validation set, for reporting)
    T = minimize_scalar(lambda T: nll(softmax(val_lg / T), y_val),
                        bounds=(0.5, 3), method="bounded").x
    np.save("temperature.npy", T)
    test_p = softmax(test_lg / T)
    pred = test_p.argmax(1); conf = test_p.max(1); correct = (pred == y_te).astype(int); n = len(y_te)

    print(f"\n=== Classification (T={T:.4f}) ===")
    print(f"accuracy={correct.mean():.4f}  macro-F1={f1_score(y_te, pred, average='macro'):.4f}")
    print(classification_report(y_te, pred, target_names=CLASS, digits=3))
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))

    print("\n=== Calibration (4 dp) ===")
    for tag, P in [("before", softmax(test_lg)), ("after ", test_p)]:
        print(f"  {tag}: ECE={ece(P, y_te):.4f}  Brier={brier(P, y_te):.4f}  NLL={nll(P, y_te):.4f}")

    rng = np.random.default_rng(0); B = 1000
    def boot(fn):
        v = [fn(rng.integers(0, n, n)) for _ in range(B)]
        return np.nanmean(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
    print("\n=== Bootstrap 95% CI ===")
    for nm, fn in [("Accuracy", lambda i: correct[i].mean()),
                   ("Macro-F1", lambda i: f1_score(y_te[i], pred[i], average="macro")),
                   ("Err-AUROC", lambda i: roc_auc_score(1 - correct[i], 1 - conf[i])
                                  if len(np.unique(correct[i])) > 1 else np.nan)]:
        m, lo, hi = boot(fn); print(f"  {nm}: {m:.4f} [{lo:.4f}, {hi:.4f}]")

    order = np.argsort(-conf); cs = correct[order]; tot = int((cs == 0).sum())
    print(f"\n=== Selective prediction / clinical utility (total errors={tot}) ===")
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = int(n * cov); removed = tot - int((cs[:k] == 0).sum())
        print(f"  coverage {int(cov*100)}%: retained_acc={cs[:k].mean():.4f} "
              f"errors_removed={removed} ({removed/tot*100:.1f}%)")

    thr = np.sort(conf)[int(0.2 * n)]
    print("\n=== Per-pattern reliability at 80% coverage ===")
    for c in range(N):
        idx = y_te == c; ret = idx & (conf >= thr)
        print(f"  {CLASS[c]:16s} acc={correct[idx].mean():.3f} conf={test_p[idx].max(1).mean():.3f} "
              f"ECE={ece(test_p[idx], y_te[idx]):.3f} defer={(conf[idx] < thr).mean():.2f} "
              f"ret_acc={(pred[ret] == y_te[ret]).mean():.3f}")

    # split conformal with an INDEPENDENT calibration set (half of validation for temperature,
    # the other half for conformal calibration) -> respects exchangeability.
    perm = rng.permutation(len(y_val)); h = len(perm) // 2; iA, iB = perm[:h], perm[h:]
    Tc = minimize_scalar(lambda T: nll(softmax(val_lg[iA] / T), y_val[iA]),
                         bounds=(0.5, 3), method="bounded").x
    confB = softmax(val_lg[iB] / Tc); yB = y_val[iB]; testc = softmax(test_lg / Tc)
    cal = 1 - confB[np.arange(len(yB)), yB]; nf = len(cal)
    def lac(a):
        qh = np.quantile(cal, np.ceil((nf + 1) * (1 - a)) / nf, method="higher"); t = 1 - qh
        s = [np.where(testc[i] >= t)[0] for i in range(len(testc))]
        return [x if len(x) > 0 else np.array([testc[i].argmax()]) for i, x in enumerate(s)]
    print(f"\n=== Split conformal (independent calibration, Tc={Tc:.3f}, calib={len(yB)}) ===")
    for a in [0.10, 0.05, 0.01]:
        s = lac(a)
        print(f"  target={1-a:.2f} coverage={np.mean([y_te[i] in s[i] for i in range(n)]):.3f} "
              f"size={np.mean([len(x) for x in s]):.3f} "
              f"singleton={np.mean([len(x) == 1 for x in s])*100:.1f}%")
    s = lac(0.01)
    print("Per-class set size @ 99% target:")
    for c in range(N):
        ix = np.where(y_te == c)[0]
        print(f"  {CLASS[c]:16s} size={np.mean([len(s[i]) for i in ix]):.3f}")


if __name__ == "__main__":
    main()
