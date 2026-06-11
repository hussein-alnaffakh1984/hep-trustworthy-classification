# HEp-2 trustworthy classification — code

Training and analysis scripts for a compact convolutional classifier of HEp-2
immunofluorescence cell images, together with the reliability components used in the
accompanying study: temperature scaling, split-conformal prediction, selective
prediction, a feature-space out-of-distribution detector, and a backbone comparison.

This repository contains source code only. Results, figures, and the manuscript are
released separately with the publication.

## Requirements

```bash
pip install -r requirements.txt
```

## Data

The scripts expect the public single-cell HEp-2 collection (Kaggle slug
`dumplinghead/hep-2-cells`), which provides `labels.mat` and the cell images. Place it
under `./data/` so that `./data/**/labels.mat` resolves, or run on Kaggle where the path
under `/kaggle/input/` is detected automatically.

## Running

Each script auto-detects the dataset and writes its artifacts to the working directory.
Run them in order:

```bash
python src/train.py             # train + calibration + selective + conformal + CIs
python src/robustness_ood.py    # corruption robustness + MSP/Energy detection
python src/mahalanobis_ood.py   # feature-space Mahalanobis detector
python src/compare_backbones.py # shallow / EfficientNet-B0 / ConvNeXt-Tiny + tests
```

`train.py` saves the model, the fitted temperature, and the logits/labels used by the
other scripts.

## Reproducibility

All scripts fix `seed = 42`. Minor variation at the fourth decimal is expected across
hardware and driver versions.

## License

MIT — see [LICENSE](LICENSE).
