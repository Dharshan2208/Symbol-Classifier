import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import time

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for consistent input sizes

set_seed()

# Enhanced GPU configuration
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Speed up training
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere GPUs
    torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on Ampere GPUs

# Multi-GPU support
def get_device():
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if n_gpu > 1:
            print(f"Found {n_gpu} GPUs! Using DataParallel")
            return torch.device('cuda'), n_gpu
        else:
            print("Using single GPU")
            return torch.device('cuda'), 1
    else:
        print("No GPU available, using CPU instead.")
        return torch.device('cpu'), 0

device, num_gpus = get_device()
print(f"Using device: {device}")

# Mixed precision training setup
use_amp = torch.cuda.is_available()  # Use mixed precision if CUDA is available
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Configuration parameters
class Config:
    batch_size = 32  # Reduced batch size for better gradient updates
    learning_rate = 0.0003  # Lower learning rate for more stable training
    num_epochs = 20
    weight_decay = 2e-5  # Slightly increased for better regularization
    num_workers = 4  # For data loading
    train_val_split = 0.9  # 90% train, 10% validation
    pin_memory = torch.cuda.is_available()  # Pin memory if CUDA is available

config = Config()

# Custom Dataset
class MathSymbolDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, is_test=False):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_path'])
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, self.df.iloc[idx]['example_id']
        else:
            label = self.df.iloc[idx]['label']
            return image, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Matched with training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Increased from 32x32 to 48x48 for more detail
    transforms.RandomRotation(20),  # Increased rotation range
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=10),  # Stronger affine transforms
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),  # More perspective distortion
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Stronger color jittering
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))  # More random erasing
])

# ResNet-like block for better model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Improved ResNet Model with deeper architecture
class ResNetMathSymbol(nn.Module):
    def __init__(self, num_classes=1401):  # Updated to match the adjusted number of classes
        super(ResNetMathSymbol, self).__init__()
        self.in_channels = 64

        # Initial layer with larger kernel for better feature detection
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Deeper residual layers
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

        # Global pooling and classifier
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)  # Increased dropout for better regularization
        self.fc = nn.Linear(512, num_classes)

        # Initialize weights for better convergence
        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, kernel_size=2, stride=2)  # Added initial pooling

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Improved CNN Model
class MathSymbolCNN(nn.Module):
    def __init__(self, num_classes=369):
        super(MathSymbolCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.3)

        # Dynamic feature size calculation for different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.bn5(F.relu(self.conv5(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.bn6(F.relu(self.fc1(x)))
        x = self.dropout4(x)
        x = self.fc2(x)
        return x

# Enhanced Mixup augmentation with more robust implementation
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    # Use device-aware index generation
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Enhanced training function with mixed precision, improved GPU utilization, and learning rate scheduling
def train(model, train_loader, criterion, optimizer, scheduler, epoch, use_mixup=True, clip_grad_norm=1.0):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Apply mixup augmentation
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
            images, labels_a, labels_b = images.to(device), labels_a.to(device), labels_b.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        if clip_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # Update weights with gradient scaling
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        train_loss += loss.item()

        if use_mixup:
            # For mixup, use argmax prediction and compare with the primary (a) labels for tracking accuracy
            _, predicted = outputs.max(1)
            total += labels_a.size(0)
            correct += (lam * predicted.eq(labels_a).sum().float() +
                        (1 - lam) * predicted.eq(labels_b).sum().float()).item()
        else:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': train_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })

    return train_loss / len(train_loader), 100. * correct / total

# Enhanced validate function with more metrics
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validating'):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_accuracy = 100. * correct / total

    # Compute per-class accuracy (useful for imbalanced datasets)
    class_correct = np.zeros(model.fc.out_features)
    class_total = np.zeros(model.fc.out_features)

    for pred, label in zip(all_preds, all_labels):
        class_correct[label] += (pred == label)
        class_total[label] += 1

    # Avoid division by zero
    class_total = np.maximum(class_total, 1)
    per_class_acc = class_correct / class_total
    avg_per_class_acc = np.mean(per_class_acc)

    # Make sure CUDA synchronizes properly
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    print(f'Validation Loss: {val_loss/len(val_loader):.4f} | Validation Accuracy: {val_accuracy:.2f}%')
    print(f'Average Per-Class Accuracy: {avg_per_class_acc*100:.2f}%')
    return val_loss / len(val_loader), val_accuracy, avg_per_class_acc*100

# Enhanced predict function with confidence scores
def predict(model, test_loader, return_probs=False):
    model.eval()
    predictions = []
    example_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc='Predicting'):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

            predictions.extend(predicted.cpu().numpy())
            example_ids.extend(ids)

            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    if return_probs:
        return example_ids, predictions, all_probs
    return example_ids, predictions

# Ensemble prediction function - allows combining multiple models for better accuracy
def ensemble_predict(models, test_loader, weights=None):
    print("Running ensemble prediction...")

    all_models_predictions = []
    example_ids = []
    processed_batches = False

    for model in models:
        model.eval()
        model_predictions = []

        if not processed_batches:
            example_ids = []

        with torch.no_grad():
            for batch_idx, (inputs, ids) in enumerate(tqdm(test_loader, desc="Predicting")):
                inputs = inputs.to(device)

                # Use mixed precision for inference
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)

                # Get class predictions
                _, predicted = torch.max(outputs, 1)

                # Store predictions from this model
                model_predictions.extend(predicted.cpu().numpy())

                # Only store example_ids once
                if not processed_batches:
                    example_ids.extend(ids)

        processed_batches = True
        all_models_predictions.append(model_predictions)

    # Convert to numpy arrays for easier manipulation
    all_models_predictions = np.array(all_models_predictions)

    # If weights are provided, use weighted average
    if weights is not None:
        weights = np.array(weights).reshape(-1, 1)
        weighted_predictions = np.sum(all_models_predictions * weights, axis=0)
        final_predictions = np.argmax(weighted_predictions, axis=1)
    else:
        # Use majority voting
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=all_models_predictions.shape[2]).argmax(),
            axis=0,
            arr=all_models_predictions
        )

    # Return example_ids and predictions
    return example_ids, final_predictions

# Main execution
def main():
    # Data loading
    print("Loading data...")

    # Replace these paths with the actual paths to your data
    train_csv = 'train.csv'  # Update with your path
    test_csv = 'test.csv'    # Update with your path
    root_dir = 'Dataset_Image'           # Update with your path

    try:
        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files. {e}")
        print("Please ensure train.csv and test.csv are in the correct location.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Check for invalid labels
    print("Validating labels...")
    min_label = train_df['label'].min()
    max_label = train_df['label'].max()

    if min_label < 0:
        print(f"Error: Found negative labels! Min label: {min_label}")
        print("Fixing negative labels by shifting all labels to be zero-indexed...")
        # Shift all labels to make them zero-indexed
        train_df['label'] = train_df['label'] - min_label
        max_label = train_df['label'].max()

    # Check number of classes
    num_classes = train_df['label'].nunique()
    print(f"Number of classes: {num_classes}")
    print(f"Label range: {min_label} to {max_label}")

    # Verify that max_label is within proper range
    if max_label >= num_classes:
        print(f"Warning: Max label ({max_label}) is >= number of unique classes ({num_classes})")
        # Set num_classes to max_label + 1 to ensure all labels are covered
        num_classes = max_label + 1
        print(f"Adjusted number of classes to: {num_classes}")

    # Class distribution analysis
    class_counts = train_df['label'].value_counts().sort_index()
    plt.figure(figsize=(15, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.savefig('class_distribution.png')
    plt.close()

    # Check for class imbalance
    min_samples = class_counts.min()
    max_samples = class_counts.max()
    print(f"Class imbalance: Min samples per class: {min_samples}, Max samples per class: {max_samples}")

    # Split train data into train and validation
    train_size = int(len(train_df) * config.train_val_split)

    # Stratified split to maintain class distribution
    from sklearn.model_selection import train_test_split
    train_df_split, val_df = train_test_split(
        train_df,
        test_size=1-config.train_val_split,
        stratify=train_df['label'],
        random_state=42
    )

    print(f"Training samples: {len(train_df_split)}, Validation samples: {len(val_df)}")

    # Create datasets
    train_dataset = MathSymbolDataset(train_df_split, root_dir, transform=train_transform)
    val_dataset = MathSymbolDataset(val_df, root_dir, transform=transform)
    test_dataset = MathSymbolDataset(test_df, root_dir, transform=transform, is_test=True)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False,
        prefetch_factor=2 if config.num_workers > 0 else None
    )

    # Create model (choose between CNN and ResNet)
    model = ResNetMathSymbol(num_classes=num_classes).to(device)

    # Enable DataParallel for multi-GPU training
    if num_gpus > 1:
        model = nn.DataParallel(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                     verbose=True, cooldown=1, min_lr=1e-6)

    best_val_acc = 0.0

    for epoch in range(config.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, epoch, use_mixup=True, clip_grad_norm=1.0)

        # Validate
        val_loss, val_acc, avg_per_class_acc = validate(model, val_loader, criterion)

        # Update scheduler based on validation accuracy
        scheduler.step(val_acc)

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{config.num_epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Avg Class Acc: {avg_per_class_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best accuracy: {best_val_acc:.2f}%. Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')

    # Load best model for prediction
    print("Loading best model for prediction...")
    model.load_state_dict(torch.load('best_model.pth'))

    # Make predictions
    print("Making predictions on test set...")
    example_ids, predictions = predict(model, test_loader)

    # Create the updated submission DataFrame with the required format: example_id and label
    # The label is the predicted class from the model
    submission = pd.DataFrame({
        'example_id': [eid.item() for eid in example_ids],
        'label': [pred.item() for pred in predictions]
    })

    submission.to_csv('submission.csv', index=False)
    print("Submission file created successfully with the correct format: example_id and label!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes")