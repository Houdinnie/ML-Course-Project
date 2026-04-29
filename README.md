# Malaria Cell Image Classification Using CNN

> **Course Project** | Nanchang Hangkong University | Computer Science

Classifies thin blood smear cell images as **Parasitized** (Plasmodium falciparum infected) or **Uninfected** using deep CNNs.

---

## Experiments

| # | Experiment | Result |
|---|---|---|
| Exp 1 | Baseline CNN (5k samples, 50 epochs) | **92.00%** |
| Exp 2 | MobileNetV2 Transfer Learning | **93.10%** |
| Exp 3 | Sample Size Reduction | Below 80% → near-random (~52%) |
| Exp 4 | Activation Functions (ReLU vs LeakyReLU vs Swish) | LeakyReLU 66.75% > ReLU 51.25% > Swish 50.00% |
| Exp 5 | Data Augmentation Impact | Comparable at 5 epochs |

---

## Dataset

- **Source**: 27,558 thin blood smear cell images (balanced: 13,779 × 2)
- **Preprocessing**: Resize to 64×64×3, normalize [0,1]
- **Split**: 80/20 stratified train-test
- **Original RAR archive**: `Machine Learning Course Project.rar`

---

## Setup

```bash
pip install numpy tensorflow scikit-learn opencv-python matplotlib
```

## Run

```bash
python run_exp4.py   # Activation functions experiment
python run_exp5.py   # Data augmentation experiment
```

> **Note**: Images are loaded from `cell_images/` folder. The dataset (`.npy` files) is excluded from this repo due to size. Download the original dataset and preprocess using the script in `Resource Code - CNN.py`.

---

## Key Findings

1. **Transfer learning provides incremental gains** — MobileNetV2 improved accuracy from 87.4% to 93.1%
2. **Sufficient data is critical** — Below 80% sample retention, accuracy drops to near-random chance
3. **Training duration matters** — Short runs (5 epochs) are insufficient for activation function differentiation
4. **LeakyReLU outperforms ReLU and Swish** on this task (with sufficient training)
