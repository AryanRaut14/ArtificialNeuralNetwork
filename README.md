# Artificial Neural Network (ANN)

A comprehensive project exploring the implementation, optimization, and hyperparameter tuning of Artificial Neural Networks (ANN) using **PyTorch** on the Fashion-MNIST dataset. This repository documents a step-by-step deep learning workflow, progressing from baseline models on small data subsets to fully optimized architectures using modern regularization techniques and automated hyperparameter searches.

---

## 📂 Repository Structure

*   **`ann_on_(fashion_mnist_6000imgs_.ipynb`**  
    The prototype phase. Implements a baseline ANN on a small subset (6,000 images) of the total dataset to quickly verify model architecture and pipeline mechanics.
*   **`ann_on_(fashion_mnist.ipynb`**  
    The full-scale implementation. Utilizes the entire Fashion-MNIST dataset scaled across hardware acceleration (GPU) for full-capacity training.
*   **`ann_optimized.ipynb`**  
    Advanced optimization script implementing crucial techniques to combat overfitting and improve generalization:
    *   **Dropout Regularization** ($p=0.3$) to prevent co-adaptation of features.
    *   **Batch Normalization (`BatchNorm1d`)** to stabilize and accelerate training.
    *   **Weight Decay ($L2$ Regularization)** via the optimizer to constrain weight magnitudes.
*   **`ann_optimized-2params.ipynb`**  
    Automated hyperparameter tuning phase. Leverages **Optuna** to run automated searches across structural parameters (`num_hidden_layer` and `neurons_per_layer`).
*   **`data/`**  
    Contains the dataset features and labels. The large primary dataset is tracked using **Git LFS (Large File Storage)** to seamlessly manage files exceeding GitHub's 100MB threshold.

---

## 🛠️ Tech Stack & Dependencies

*   **Framework:** PyTorch (Core deep learning backend)
*   **Optimization Framework:** Optuna (Automated hyperparameter tuning)
*   **Data Processing:** Pandas, NumPy, Scikit-Learn
*   **Storage Management:** Git LFS (Large File Storage)

---

