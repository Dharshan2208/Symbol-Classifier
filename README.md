# 🔥 Torch it Up - ML Challenge: Handwritten Mathematical Symbol Classification

Welcome to the **Handwritten Mathematical Symbol Classification Challenge**! This competition invites you to design and train a robust machine learning model to classify handwritten mathematical symbols with high accuracy. Handwritten symbols are essential in applications like optical character recognition (OCR), digital note-taking, and assistive technologies. However, variations in writing styles, distortions, and noise make this a fascinating and complex task.

In this challenge, you'll work with a dataset of **168,236 grayscale images (32×32 pixels)** spanning **369 unique classes** of mathematical notation. Your goal is to develop a model using the labeled training data and generate predictions for an unlabeled test set, aiming to maximize accuracy on the private leaderboard.

This repository provides a **PyTorch-based solution** featuring a deep learning pipeline with advanced techniques like residual networks (ResNet), mixed precision training, data augmentation, and multi-GPU support. Let’s dive in and torch it up! 🔥

---

## 📊 Dataset Overview

- **Training Set**: Labeled dataset with 168,236 grayscale images (32×32 pixels) across 369 classes.
- **Test Set**: Unlabeled dataset for submitting predictions.
- **Image Format**: Grayscale (single-channel), resized to 48×48 pixels during preprocessing.
- **Classes**: 369 unique mathematical symbols (e.g., digits, operators, Greek letters, etc.).
- **Files**:
  - `train.csv`: Contains `image_path` and `label` columns.
  - `test.csv`: Contains `example_id` and `image_path` columns.
  - `Dataset_Image/`: Directory with image files.

---

## 🚀 Features of This Solution

This codebase implements a state-of-the-art deep learning pipeline optimized for performance and accuracy:

- **Model Architecture**:
  - Custom **ResNetMathSymbol**: A deep residual network with 4 residual layers (64, 128, 256, 512 filters), batch normalization, dropout, and adaptive pooling.
  - Alternative: **MathSymbolCNN**: A lighter convolutional neural network with 5 conv layers and adaptive pooling.
- **Data Augmentation**: Robust preprocessing with random rotations, affine transforms, perspective distortion, color jittering, and random erasing.
- **Training Enhancements**:
  - **Mixed Precision Training**: Leverages NVIDIA’s AMP for faster computation.
  - **Multi-GPU Support**: Uses `DataParallel` for parallel training on multiple GPUs.
  - **Mixup Augmentation**: Blends samples and labels to improve generalization.
  - **Gradient Clipping**: Stabilizes training with a max norm of 1.0.
  - **Learning Rate Scheduling**: Reduces LR on plateau to fine-tune convergence.
- **Evaluation Metrics**: Tracks validation loss, accuracy, and per-class accuracy for imbalanced data insights.
- **Prediction Pipeline**: Generates submission-ready predictions with `example_id` and `label` columns.

---

## 🛠️ Requirements

To run this code, install the following dependencies:

```bash
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow tqdm
```

- **Hardware**: GPU recommended (NVIDIA CUDA-enabled for mixed precision and multi-GPU support).
- **Python**: 3.8+ recommended.

---

## 📂 Project Structure

```
├── train.csv              # Training data with labels
├── test.csv               # Test data for predictions
├── Dataset_Image/         # Directory containing image files
├── script.py              # Main script (provided code)
├── best_model.pth         # Saved best model weights (generated)
├── submission.csv         # Output predictions (generated)
├── class_distribution.png # Class distribution plot (generated)
└── README.md              # This file
```

---

## 🏃‍♂️ How to Run

1. **Prepare the Data**:
   - Place `train.csv`, `test.csv`, and the `Dataset_Image/` folder in the working directory.
   - Update the paths in the `main()` function if necessary:
     ```python
     train_csv = 'path/to/train.csv'
     test_csv = 'path/to/test.csv'
     root_dir = 'path/to/Dataset_Image'
     ```

2. **Run the Script**:
   ```bash
   python script.py
   ```
   - The script trains the model, validates it, and generates `submission.csv`.
   - Training progress is displayed with a progress bar (`tqdm`).

3. **Output**:
   - `best_model.pth`: Best model weights based on validation accuracy.
   - `submission.csv`: Predictions in the required format (`example_id`, `label`).
   - `class_distribution.png`: Visualization of class distribution in the training set.

---

## 🔧 Configuration

Key hyperparameters are defined in the `Config` class and can be tuned:

```python
class Config:
    batch_size = 32          # Batch size for training
    learning_rate = 0.0003   # Initial learning rate
    num_epochs = 20          # Number of training epochs
    weight_decay = 2e-5      # L2 regularization strength
    num_workers = 4          # DataLoader workers
    train_val_split = 0.9    # Train/validation split ratio
```

Adjust these values based on your hardware and experimentation needs.

---

## 📈 Performance Tips

- **Class Imbalance**: The code analyzes class distribution and uses stratified splitting to maintain balance in train/validation sets.
- **Overfitting**: Dropout (up to 0.5) and weight decay (2e-5) are applied; consider increasing these for smaller datasets.
- **Hardware Utilization**: Enable `pin_memory=True` and adjust `num_workers` for faster data loading on GPUs.
- **Model Selection**: Switch to `MathSymbolCNN` for a lighter model if compute resources are limited:
  ```python
  model = MathSymbolCNN(num_classes=num_classes).to(device)
  ```

---

## 📝 Submission Format

The output `submission.csv` follows the competition format:

```
example_id,label
0,5
1,42
2,17
...
```

Each row maps a test sample’s `example_id` to its predicted `label`.

---

## 🌟 Future Improvements

- **Ensemble Learning**: Use the `ensemble_predict` function to combine multiple models for higher accuracy.
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and augmentation strength.
- **Data Expansion**: Incorporate external datasets of mathematical symbols to boost robustness.
- **Advanced Architectures**: Try Vision Transformers (ViT) or EfficientNet for potential gains.

---

## ⏱️ Execution Time

The script logs total execution time. On a single NVIDIA GPU (e.g., RTX 3090), training for 20 epochs takes approximately **30-50 minutes**, depending on dataset size and hardware.

---
