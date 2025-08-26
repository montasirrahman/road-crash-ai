# Road-Crash-AI 🚗💥 (Video Accident Detection)

A TensorFlow/Keras project for **accident vs. non-accident video classification** using a **TimeDistributed CNN + LSTM** pipeline. 
It loads short video clips, extracts frames, trains a sequence model, and provides evaluation plots (loss/accuracy, confusion matrix, ROC, PR curve, threshold tuning, and error analysis). It also includes a **real-time style video tester** that overlays predictions on frames.

> **File:** `road-crash-ai.py`  
> **Frameworks:** TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Seaborn, scikit-learn

---

## 🔧 Features

- Frame extraction from videos (resizing + normalization)
- Sequence model: `Conv2D` (TimeDistributed) → `BatchNorm` → `MaxPool` → `Flatten` → `LSTM` → `Dense`
- Trains and validates with dataset split (`./dataset/train`, `./dataset/test`)
- Saves the **best model** to `./models/video_model.h5`
- Evaluation:
  - Loss & Accuracy curves
  - Confusion Matrix (with labels)
  - ROC Curve + AUC
  - Precision–Recall Curve + Average Precision
  - Class distribution
  - Threshold vs F1-score optimization
  - Error analysis (False Positives & False Negatives frame samples)
- Batch testing on arbitrary videos in `./videos` with overlay text

---

## 🗂️ Project Structure

```
.
├── road-crash-ai.py
├── dataset
│   ├── train
│   │   ├── Accident
│   │   └── Non-Accident
│   └── test
│       ├── Accident
│       └── Non-Accident
├── models
│   └── video_model.h5        # auto-created
├── videos/                   # put your test videos here
└── figures/                  # optional: save evaluation plots here
```

> The code expects **binary classes** named exactly `Accident` and `Non-Accident` in both `train/` and `test/`.

---

## 📦 Requirements

- Python 3.9–3.11 (recommended)
- TensorFlow 2.10+ (CPU or GPU)
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

### Install (Conda example)

```bash
# Create and activate env
conda create -n roadcrash python=3.10 -y
conda activate roadcrash

# TensorFlow (CPU)
pip install "tensorflow>=2.10,<2.17"

# Or TensorFlow (GPU) — choose a build compatible with your CUDA/cuDNN
# pip install "tensorflow[and-cuda]>=2.13,<2.17"

# Core deps
pip install opencv-python numpy matplotlib seaborn scikit-learn
```

> **Windows tip:** If you see `A module that was compiled using NumPy 1.x cannot be ...` then pin versions to compatible pairs, e.g.:
>
> ```bash
> pip install "numpy<2.0"  # for older TF wheels expecting NumPy 1.x
> ```
>
> Or upgrade TF to a version compatible with NumPy 2.x.

---

## 📚 How the Dataset Should Look

Place videos in these folders:

```
dataset/
  train/
    Accident/
      crash001.mp4
      ...
    Non-Accident/
      normal001.mp4
      ...
  test/
    Accident/
      crash101.mp4
      ...
    Non-Accident/
      normal101.mp4
      ...
```

The script **auto-labels** classes by folder name and **expects** all videos to be readable by OpenCV.

---

## 🚀 Run Training & Evaluation

```bash
python road-crash-ai.py
```

What happens:
1. Loads videos from `dataset/train` and `dataset/test`
2. Trains for `EPOCHS=15` with `BATCH_SIZE=8`
3. Saves best model to `models/video_model.h5` (via `ModelCheckpoint`)
4. Displays evaluation plots
5. Iterates through `./videos` and runs the predictor with overlay text

> Press **`q`** while the prediction window is focused to stop the video loop.

---

## 🧠 Model Details

- **Sequence length:** `SEQ_LENGTH = 15` frames (uniformly sampled)
- **Frame size:** `224x224`, normalized to `[0,1]`
- **Architecture:**  
  `TimeDistributed(Conv2D→BN→MaxPool) × 2 → TimeDistributed(Flatten) → LSTM(64) → Dense(1, sigmoid)`  
- **Loss:** Binary cross-entropy, **Optimizer:** Adam, **Metric:** Accuracy

---

## 📈 Figures & Saving Plots

The script already builds these plots:
- Loss vs. Epochs, Accuracy vs. Epochs
- Confusion Matrix
- ROC (AUC)
- Precision–Recall (Average Precision)
- Class distribution
- F1 vs. Threshold
- False Positives / False Negatives (sample frames)

To **save all figures** to `./figures`, add this snippet **before** `plt.show()`:

```python
import os
os.makedirs("./figures", exist_ok=True)
for i, fig in enumerate(map(plt.figure, plt.get_fignums())):
    fig.savefig(f"./figures/figure_{i+1}.png", dpi=200, bbox_inches="tight")
```

> If you run on a headless server, set a non-interactive backend at the very top _before_ importing pyplot:
>
> ```python
> import matplotlib
> matplotlib.use("Agg")
> import matplotlib.pyplot as plt
> ```

---

## 🧪 Testing on Your Own Videos

Drop any `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv` files into `./videos`.  
The script will call:

```python
test_new_video(model, video_path, frames_per_second=5, sequence_length=SEQ_LENGTH)
```

- It samples frames at `frames_per_second`
- Predicts every `sequence_length` frames
- Overlays label: **Accident** or **No Accident** with confidence

---

## ⚠️ Troubleshooting

- **NumPy / TensorFlow ABI mismatch (common on Windows):**  
  - Try `pip install "numpy<2.0"` or upgrade TensorFlow to a wheel that supports NumPy 2.x.
  - Ensure only one Python environment is active (avoid mixing Conda + system Python).
- **OpenCV video read fails:**  
  - Print `cv2.getBuildInformation()` to ensure FFMPEG is enabled.  
  - Try re-encoding videos to H.264 `.mp4`.
- **GPU not used:**  
  - Install CUDA/cuDNN versions matching your TensorFlow version.  
  - Verify with `tf.config.list_physical_devices('GPU')`.
- **Class folders missing or empty:**  
  - Ensure both `Accident` and `Non-Accident` exist in `train/` and `test/`.
- **Imbalanced data:**  
  - Consider class weighting or data augmentation for under-represented class.

---

## 🔁 Customization

- Change sequence length: `SEQ_LENGTH = 15`
- Input size: `IMG_HEIGHT, IMG_WIDTH = 224, 224`
- Training loops: `EPOCHS`, `BATCH_SIZE`
- Add more `Conv2D` blocks or increase `LSTM` units for capacity
- Replace backend with 3D CNNs (e.g., `Conv3D`) for spatiotemporal features (advanced)

---

## 📜 License

MIT License. Feel free to use and modify for research/education.

---

## 🙌 Acknowledgements

- TensorFlow/Keras for deep learning
- OpenCV for video I/O
- scikit-learn & Matplotlib/Seaborn for evaluation and plots

---

## 💡 Next Steps (Ideas)

- Hard-mining clips near decision boundary (improve F1)
- Use pretrained CNN backbones (e.g., MobileNetV2) inside `TimeDistributed`
- 3D CNNs (C3D/I3D) or Transformers for video understanding
- Temporal augmentation: varied sampling rates; random start offsets
- Export to ONNX/TFLite for edge deployment
