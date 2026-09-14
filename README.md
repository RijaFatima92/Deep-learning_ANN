```markdown
# Deep Learning for Perception — Assignment 1 (Fall 2026)
**Department of Computer Science, National University of Computer and Emerging Sciences (FAST-NUCES)**[cite: 1]

---

## 📌 Project Overview
This repository contains the complete implementation, experimental evaluation, and analysis for **Deep Learning for Perception — Assignment 1 (Fashion-MNIST)**[cite: 1]. 

The project investigates foundational deep learning concepts, progressing from backpropagation implemented from scratch in pure NumPy[cite: 1] to architectural baseline comparisons, activation function studies, loss dynamics, optimizer benchmarks, regularization methods, and convolutional neural network (CNN) architectures[cite: 1].

---

## 🎯 Final Benchmark Results
* **Final Model Architecture:** `FashionCNN` (2D Convolutions + Batch Normalization + Max-Pooling + Dropout + Dense Head)[cite: 1]
* **Holdout Test Accuracy:** **91.82%**[cite: 1]
* **Holdout Test Loss:** **0.2696** (Categorical Cross-Entropy)[cite: 1]
* **Validation Accuracy:** **91.82%**[cite: 1]
* **Best Dense MLP Accuracy:** **89.11%** (under combined Batch Normalization, Dropout $p=0.2$, and $L_2$ decay)[cite: 1]
* **Single Change That Helped Most:** Transitioning from 1D flattened dense Multi-Layer Perceptrons to 2D Spatial Convolutions with hierarchical pooling[cite: 1], yielding a **+2.71%** absolute test accuracy gain by preserving spatial pixel hierarchies and translation invariance[cite: 1].

---

## 📁 Repository Structure
```text
├── DL_ASS01_23F-0725_23F-0781.ipynb   # Complete Jupyter Notebook with all outputs & plots
├── fashion_cnn_best.pth               # Saved PyTorch weights for optimal FashionCNN model
├── requirements.txt                   # Environment dependencies
└── README.md                          # Reproduction guide and documentation
```[cite: 1]

---

## 🔬 Experimental Progression Summary

1. **Part 1: Backpropagation from Scratch (NumPy Only)**
   * Implemented a $784 \to 64 \to 10$ MLP using only NumPy vector operations, ReLU activations, and softmax cross-entropy[cite: 1].
   * Evaluated via analytical PyTorch autograd gradient checking with a batch size of $N=128$[cite: 1].
   * **Gradient Difference:** Maximum absolute difference between NumPy manual gradients and PyTorch autograd was $\mathbf{5.55 \times 10^{-17}}$, confirming exact mathematical correctness[cite: 1].

2. **Part 2: Baseline Model & Activation Study**
   * Benchmarked identical $784 \to 128 \to 64 \to 10$ networks across **Sigmoid**, **Tanh**, **ReLU**, and **LeakyReLU** ($\alpha=0.01$)[cite: 1].
   * Gradient monitoring on the first layer demonstrated vanishing gradients for Sigmoid (initial mean gradient: $0.000098$) compared to ReLU ($0.001268$)[cite: 1].
   * Observed a dead hidden ReLU unit rate of **2.08%** across hidden layers on validation batches[cite: 1].

3. **Part 3: Loss Function Formulation & Tabular Regression**
   * Demonstrated that Categorical Cross-Entropy converges significantly faster and achieves superior accuracy (**87.38%**) compared to Mean Squared Error on one-hot targets (**76.49%**)[cite: 1].
   * Trained an auxiliary MLP on the Diabetes tabular regression dataset, reporting test metrics: **MSE: 2829.65**, **RMSE: 53.19**, and **MAE: 43.21**[cite: 1].

4. **Part 4: Regularization & Overfitting Mitigation**
   * Systematically analyzed unregularized baselines against $L_2$ weight decay ($1\times 10^{-3}$), Dropout ($p=0.3$), Batch Normalization, and a combined regularizer[cite: 1]:
     * *Unregularized Baseline:* 92.32% Train | 88.28% Val (4.04% Gap)[cite: 1]
     * *Dropout ($p=0.3$):* 89.01% Train | 88.48% Val (0.54% Gap)[cite: 1]
     * *Combined (BN + Dropout + $L_2$):* 91.24% Train | 89.11% Val (2.13% Gap)[cite: 1]

5. **Part 5: FashionCNN Architecture & Evaluation**
   * Designed a custom 2D CNN with two convolutional stages (`Conv2D-BN-ReLU-MaxPool`) followed by dropout and linear projection[cite: 1].
   * Surpassed dense architectures, achieving **91.82% test accuracy** with high precision across distinct garment and footwear categories[cite: 1].

---

## 🛠️ Environment Setup & Installation

### Prerequisites
* Python 3.10+ (tested on Python 3.12)[cite: 1]
* CUDA-compatible GPU recommended (tested on NVIDIA Tesla T4 $\times 2$)[cite: 1]

### Installation
1. Clone this repository:
   ```bash
   git clone [https://github.com/RijaFatima/FashionMNIST-DeepLearning-Perception.git](https://github.com/RijaFatima/FashionMNIST-DeepLearning-Perception.git)
   cd FashionMNIST-DeepLearning-Perception

```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

```


3. Install dependencies:
```bash
pip install -r requirements.txt

```



---

## 🚀 How to Reproduce Results

### 1. Dataset Setup

The notebook is configured to run seamlessly on Kaggle or local environments.

* **Kaggle:** Add the `zalando-research/fashionmnist` dataset (`fashion-mnist_train.csv` and `fashion-mnist_test.csv`) to your session inputs.


* **Local Run:** Place `fashion-mnist_train.csv` and `fashion-mnist_test.csv` in the root directory. The data loader automatically detects local paths.



### 2. Run the Full Notebook

Execute all cells sequentially via Jupyter Notebook or VS Code:

```bash
jupyter notebook DL_ASS01_23F-0725_23F-0781.ipynb

```

All stochastic operations, data splits, and model initializations are anchored to `SEED = 42` for strict determinism:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```[cite: 1]

### 3. Standalone Inference with Saved Model
To run inference on a single image using the saved weights (`fashion_cnn_best.pth`)[cite: 1]:

```python
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Define architecture matching saved weights
from your_module import FashionCNN, predict_single_image

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Predict
pred_label, confidence = predict_single_image(
    model_path="fashion_cnn_best.pth",
    image_tensor=your_image_tensor,
    class_names=class_names
)
print(f"Predicted: {pred_label} ({confidence:.2f}%)")
```[cite: 1]

---

## 📊 Class-Level Performance Summary (FashionCNN)

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **T-shirt/top** | 82.53% | 88.58% | 85.45% |[cite: 1]
| **Trouser** | 98.67% | 99.00% | 98.84% |[cite: 1]
| **Pullover** | 91.79% | 83.83% | 87.63% |[cite: 1]
| **Dress** | 93.70% | 88.00% | 90.76% |[cite: 1]
| **Coat** | 87.77% | 88.50% | 88.13% |[cite: 1]
| **Sandal** | 98.91% | 98.58% | 98.75% |[cite: 1]
| **Shirt** | 73.92% | 79.58% | 76.65% |[cite: 1]
| **Sneaker** | 96.20% | 97.08% | 96.64% |[cite: 1]
| **Bag** | 99.49% | 98.00% | 98.74% |[cite: 1]
| **Ankle boot** | 97.49% | 97.00% | 97.24% |[cite: 1]

---

## 👥 Contributors
* **Student IDs:** `23F-0725`, `23F-0781`
* **Course:** Deep Learning for Perception (CS Department, FAST-NUCES)[cite: 1]

```
