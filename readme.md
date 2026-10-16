# Satellite Land Cover Classification & Change Detection

Detecting deforestation and urban expansion using transfer learning on Sentinel-2 satellite imagery. A ResNet50 model fine-tuned on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset classifies land cover across 10 classes, then compares two time periods to flag land-cover transitions at three geographically distinct study sites.

---

## Results

| Study Area | Period | Flagged Transitions |
|---|---|---|
| Paraná, Brazil | 2018 → 2024 | 28 patches |
| Mato Grosso, Brazil | 2018 → 2026 | 10 patches |
| Karachi, Pakistan | 2018 → 2024 | 6 patches |

Full per-class metrics, confusion matrices, and change maps are in [`results/`](results/).

---

## How It Works

```
Sentinel-2 imagery (two dates)
        │
        ▼
Geometric alignment (rasterio)       ← ensures pixel-to-pixel correspondence
        │
        ▼
64×64 px patch extraction            ← matches EuroSAT's native footprint (~640m²)
        │
        ▼
Resize to 224×224 + ImageNet norm    ← ResNet50 input requirements
        │
        ▼
ResNet50 (fine-tuned on EuroSAT)     ← 97.67% test accuracy, 10 land-cover classes
        │
        ▼
Per-patch land-cover label (×2 dates)
        │
        ▼
Change detection: Forest → {AnnualCrop, Pasture, Industrial, Residential}
```

---

## Model

- **Architecture:** ResNet50 pretrained on ImageNet
- **Dataset:** EuroSAT RGB (27,000 images, 10 classes, 64×64 px)
- **Training strategy:** Two-phase transfer learning
  - Phase 1 — freeze all layers, train final classification head only (lr=1e-3, up to 10 epochs)
  - Phase 2 — unfreeze all layers, fine-tune end-to-end (lr=1e-4, up to 10 epochs)
- **Early stopping:** patience=3 epochs on validation accuracy
- **Best checkpoint:** saved after each improvement, reloaded before test evaluation

### Test Set Performance (10-class, 2,700 images)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| AnnualCrop | 0.97 | 0.99 | 0.98 |
| Forest | 0.99 | 0.99 | 0.99 |
| HerbVeg | 0.98 | 0.96 | 0.97 |
| Highway | 0.96 | 0.99 | 0.97 |
| Industrial | 0.96 | 0.98 | 0.97 |
| Pasture | 0.97 | 0.98 | 0.98 |
| PermCrop | 0.96 | 0.97 | 0.96 |
| Residential | 1.00 | 0.97 | 0.98 |
| River | 0.99 | 0.95 | 0.97 |
| SeaLake | 0.99 | 0.99 | 0.99 |
| **Overall** | **0.98** | **0.98** | **0.98** |

**Test accuracy: 97.67%**

---

## Notebooks

| Notebook | Description |
|---|---|
| [`notebooks/Pipeline.ipynb`](notebooks/Pipeline.ipynb) | EuroSAT download, preprocessing, model training, evaluation |
| [`notebooks/Inference.ipynb`](notebooks/Inference.ipynb) | Load trained model, classify new Sentinel-2 scenes, detect change |

Both notebooks are designed for **Google Colab** with Google Drive persistence. See [Setup](#setup) below.

---

## Repository Structure

```
├── notebooks/
│   ├── Pipeline.ipynb                    # Training pipeline
│   └── Inference.ipynb        # Inference + change detection
│
├── results/
│   ├── analysis_report.pdf
│   ├── karachi/
│   │   ├── change_detection_results(karachi).png
│   │   └── deforestation_events(karachi).csv
│   ├── mato_grosso/
│   │   ├── change_detection_results(Mato Grosso).png
│   │   └── deforestation_events(Mato Grosso).csv
│   └── parana/
│       ├── change_detection_results(Paraná).png
│       └── deforestation_events(Paraná).csv
│
├── .gitignore
│
└── README.md
```

---

## Setup

### Requirements

```bash
pip install torch torchvision rasterio scikit-learn matplotlib seaborn numpy pillow
```

### Google Colab (recommended)

Both notebooks mount Google Drive on first run. Drive is used for:
- EuroSAT dataset (~2GB, downloaded once via `torchvision.datasets.EuroSAT`)
- Model checkpoints (`best_model_phase1.pth`, `best_model_finetuned.pth`)
- Sentinel-2 input scenes and output results

**To reproduce from scratch:**
1. Open `Pipeline.ipynb` in Colab. Run all cells top to bottom. Training takes ~30–60 minutes on a T4 GPU (free Colab tier).
2. Open `Inference.ipynb` in a separate Colab session. Point `SCENE_2018` and `SCENE_2024_RAW` at your own Sentinel-2 GeoTIFF files. Run all cells.

### Sentinel-2 Imagery

Imagery was downloaded from [Copernicus Browser](https://browser.dataspace.copernicus.eu/) as:
- Product: Sentinel-2 L2A, True Color
- Format: TIFF (16-bit), WGS 84 (EPSG:4326)
- Resolution: MEDIUM (1202×1250 px)

Both time periods for each study area were downloaded with **identical settings** to ensure radiometric consistency across dates.

---

## Study Areas

### Paraná, Brazil — 2018 vs 2024
Agricultural frontier zone in southern Brazil. The Atlantic Forest biome in Paraná has faced decades of agricultural conversion; the model detects ongoing Forest → AnnualCrop/Pasture transitions.

### Mato Grosso, Brazil — 2018 vs 2026
Core deforestation hotspot in the Brazilian Amazon. Mato Grosso has historically been the state with the highest absolute deforestation rates in Brazil, driven by soy and cattle expansion.

### Karachi, Pakistan — 2018 vs 2024
Rapidly expanding coastal megacity. The model flags Forest/HerbVeg → Residential/Industrial transitions consistent with urban sprawl at the city's peri-urban fringes.

---

## Limitations

- **Patch resolution:** each 64×64 patch covers roughly 640m × 640m at 10m/pixel Sentinel-2 native resolution. Sub-patch transitions are invisible to the model.
- **Single-image time points:** no cloud masking or seasonal correction applied; phenological differences between acquisition dates (different months/seasons) can cause false-positive change flags.
- **Domain shift:** the model was trained on EuroSAT imagery (European landscapes). Performance on tropical (Brazil) and South Asian (Pakistan) scenes may differ from the reported 97.67% test accuracy.
- **True Color input:** imagery was exported as a visual True Color render rather than raw reflectance bands. This removes some radiometric precision compared to using raw band exports through Earth Engine.

---

## References

- Helber, P. et al. (2019). [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://doi.org/10.1109/JSTARS.2019.2918242). *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.*
- He, K. et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *CVPR 2016.*
- Copernicus Sentinel-2 data (2018–2026), processed by ESA.