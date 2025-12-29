#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Handle your nested folder structure
DATA_ROOT =r"C:\Users\hp\Downloads\code_move\room_design_custom"
BATCH_SIZE = 16  # Reduced batch size for CPU (easier on RAM)
NUM_EPOCHS = 7  # Reduced epochs (CPU is slower, so we start small)
LEARNING_RATE = 0.001

# FORCE CPU
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# ==========================================
# 2. DATA LOADING & TRANSFORMS
# ==========================================
# Standardize images for the model
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("\n--- Loading Data ---")
image_datasets = {}
dataloaders = {}
dataset_sizes = {}

try:
    for x in ['train', 'val']:
        path = os.path.join(DATA_ROOT, x)
        if not os.path.exists(path):
            print(f"❌ Error: Path not found: {path}")
            print("Please check your folder structure matches: my_custom_dataset/my_custom_dataset/train")
            exit()
            
        image_datasets[x] = datasets.ImageFolder(path, data_transforms[x])
        # num_workers=0 is often safer/more stable on Windows CPU
        dataloaders[x] = torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        dataset_sizes[x] = len(image_datasets[x])

    class_names = image_datasets['train'].classes
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# ==========================================
# 3. ANALYZE DATASET
# ==========================================
print(f"\n--- Dataset Analysis ---")
print(f"Classes Detected: {class_names}")
print(f"Total Training Images: {dataset_sizes['train']}")
print(f"Total Validation Images: {dataset_sizes['val']}")

# ==========================================
# 4. MODEL SETUP (ResNet50)
# ==========================================
print("\n--- Setting up ResNet50 ---")

# Load pre-trained ResNet50
# Note: 'weights' parameter is the modern way to load pretrained weights
model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers (so we don't retrain the whole brain, just the last part)
for param in model_ft.parameters():
    param.requires_grad = False

# Replace the final layer (Fully Connected) with our number of classes
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# Only optimize parameters of the final layer
optimizer_ft = optim.SGD(model_ft.fc.parameters(), lr=LEARNING_RATE, momentum=0.9)

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()
    
    # Store history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            # Enumerate helps track progress
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best one so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# Run Training
print("\n--- Starting Training (This may take a few minutes on CPU) ---")
model_ft, history = train_model(model_ft, criterion, optimizer_ft, num_epochs=NUM_EPOCHS)

# ==========================================
# 6. PLOTTING RESULTS
# ==========================================
print("\n--- Plotting Results ---")
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png') 
plt.show()

# ==========================================
# 7. SAVE MODEL
# ==========================================
save_path = 'room_style_resnet50.pth'
torch.save(model_ft.state_dict(), save_path)
print(f"\n✅ Model saved to {save_path}")


# In[ ]:




