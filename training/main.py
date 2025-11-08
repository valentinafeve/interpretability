import logging
import random
import time

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import (Compose, Normalize, Resize, ToPILImage,
                                    ToTensor)
from tqdm import tqdm, trange

from training.icnn import generate_templates
from training.loss import mutual_information
from training.model import Model
# from training.model_icnn import Model
from training.utils.plots import plot_tensors
import os

from torchmetrics import Specificity, Accuracy, Precision, Recall

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_INTERPRETABLE = True

## Hyperparameters for training
BATCH_SIZE = 5
EPOCHS = 30
DESIRED_IMAGE_SHAPE = 64
LAMBDA_ = 1e-3

if IS_INTERPRETABLE:
    EXPERIMENT_NAME = "interpretable-cnn"
    MODEL_NAME = "interpretable-cnn-model"
else:
    EXPERIMENT_NAME = "cnn"
    MODEL_NAME = "cnn-model"

CURRENT_DATE = time.strftime("%Y%m%d-%H%M%S")

CHECKPOINT_EVERY = 1
RUN_NAME = f"run_{CURRENT_DATE}"
RUN_PATH = f"runs/run_{CURRENT_DATE}"
CHECKPOINT_PATH = f"{RUN_PATH}/model_icnn_checkpoint_{CURRENT_DATE}.pth"

# random_ids_for_samples = [0, 42, 7, 13, 25, 33, 49, 58, 67, 72]
random_ids_for_samples = random.sample(range(0, 12), 10)
os.mkdir(RUN_PATH)

logger.info(f"Using interpretable model: {IS_INTERPRETABLE}")
logger.info(f"Starting training with hyperparameters:" f"lambda={LAMBDA_}, epochs={EPOCHS}, image_shape={DESIRED_IMAGE_SHAPE}")

## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f"Using device: {device}")

mlflow.set_experiment(EXPERIMENT_NAME)

transform = transforms.Compose(
    [ 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((DESIRED_IMAGE_SHAPE, DESIRED_IMAGE_SHAPE)),
        transforms.CenterCrop((DESIRED_IMAGE_SHAPE, DESIRED_IMAGE_SHAPE)),
    ]
)

## Load dataset

dataset = load_dataset("Hemg/Melanoma-Cancer-Image-Dataset", split="train")
# Sample 10% of the dataset for faster training
dataset = dataset.shuffle(seed=42).select(range(int(0.01 * len(dataset))))
trainset, testevalset = dataset.train_test_split(test_size=0.2, seed=42).values()
testset, valset = testevalset.train_test_split(test_size=0.5, seed=42).values()

trainset = trainset.with_format("torch")
testset  = testset.with_format("torch")
valset  = valset.with_format("torch")

trainset = trainset.with_transform(lambda examples: {'image': [transform(img) for img in examples['image']], 'label': examples['label']})
testset = testset.with_transform(lambda examples: {'image': [transform(img) for img in examples['image']], 'label': examples['label']})
valset = valset.with_transform(lambda examples: {'image': [transform(img) for img in examples['image']], 'label': examples['label']})

classes = trainset.features['label'].names
logger.info(f"Classes: {classes}")

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)
valloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0)

input_channels = trainset[0]['image'].shape[-3]

n = trainset[0]['image'].shape[-1]

losses = {
    "total_loss": [],
    "class_loss": [],
    "interp_loss": []
}

# Metrics
specificity = Specificity(task="multiclass", average='macro', num_classes=len(classes)).to(device)
accuracy = Accuracy(task="multiclass", average='macro', num_classes=len(classes)).to(device)
precision = Precision(task="multiclass", average='macro', num_classes=len(classes)).to(device)
recall = Recall(task="multiclass", average='macro', num_classes=len(classes)).to(device)


model = Model(feature_map_size=n, channels=3, num_classes=len(classes), filters_icnn=10)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.log_param("lambda", LAMBDA_)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("feature_map_size", n)
    
    templates = generate_templates(16, tau=0.5/(16*16), beta=4.0).to(device)
    # Set templates as not trainable
    templates.requires_grad = False
    
    for epoch in trange(EPOCHS):

        losses_epoch = {
            "total_loss": [],
            "class_loss": [],
            "interp_loss": []
        }

        for batch in tqdm(trainloader, desc="Training", leave=False):
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits, icnn_output = model(images)

            class_loss = criterion(logits, labels.long())
       
            if IS_INTERPRETABLE:
                interp_loss = mutual_information(icnn_output, templates, icnn_output.shape[-1], len(classes), labels, device=device).sum()
                total_loss = LAMBDA_ * interp_loss + (1 - LAMBDA_) * class_loss
                # total_loss = class_loss
                losses_epoch["interp_loss"].append(interp_loss.item())

            else:
                total_loss = class_loss

            losses_epoch["total_loss"].append(total_loss.item())
            losses_epoch["class_loss"].append(class_loss.item())
    
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            predicted_all = []
            labels_all = []

            for batch in valloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs, icnn_ = model(images)
                _, predicted = torch.max(outputs.data, 1)

                predicted_all.append(predicted)
                labels_all.append(labels)

            predicted_all = torch.cat(predicted_all)
            labels_all = torch.cat(labels_all)
            
            average_specificity = specificity(predicted_all, labels_all.long()).item()
            average_accuracy = accuracy(predicted_all, labels_all.long()).item()
            average_precision = precision(predicted_all, labels_all.long()).item()
            average_recall = recall(predicted_all, labels_all.long()).item()

            mlflow.log_metric("val_specificity", average_specificity, step=epoch)
            mlflow.log_metric("val_accuracy", average_accuracy, step=epoch)
            mlflow.log_metric("val_precision", average_precision, step=epoch)
            mlflow.log_metric("val_recall", average_recall, step=epoch)

            predicted_all = []
            labels_all = []
            for batch in trainloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs, icnn_ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                predicted_all.append(predicted)
                labels_all.append(labels)
            predicted_all = torch.cat(predicted_all)
            labels_all = torch.cat(labels_all)

            average_specificity = specificity(predicted_all, labels_all.long()).item()
            average_accuracy = accuracy(predicted_all, labels_all.long()).item()
            average_precision = precision(predicted_all, labels_all.long()).item()
            average_recall = recall(predicted_all, labels_all.long()).item()

            mlflow.log_metric("train_specificity", average_specificity, step=epoch)
            mlflow.log_metric("train_accuracy", average_accuracy, step=epoch)
            mlflow.log_metric("train_precision", average_precision, step=epoch)
            mlflow.log_metric("train_recall", average_recall, step=epoch)
            

        for key in losses_epoch:
            mlflow.log_metric(f"train_{key}", np.mean(losses_epoch[key]), step=epoch)

        if (((epoch + 1) % CHECKPOINT_EVERY) == 0) or (epoch == (EPOCHS - 1)):
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            mlflow.log_artifact(CHECKPOINT_PATH)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            save_path = f"{RUN_PATH}/icnn_filters_{CURRENT_DATE}_epoch{epoch}.png"

            valset_plots = torch.stack([valset[i]['image'] for i in random_ids_for_samples]).to(device)
            _, icnn_output = model(valset_plots)

            img = plot_tensors(icnn_output, save_path=save_path)
            mlflow.log_image(img, f"icnn_filters_epoch_{epoch}.png") 
            print("debug")
            if epoch == EPOCHS - 1:
                mlflow.pytorch.log_model(model, artifact_path=f"{RUN_NAME.replace('-', '_')}_model", registered_model_name=MODEL_NAME)

    # Evaluate
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in testloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs, icnn_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        mlflow.log_metric("test_accuracy", 100 * correct / total)
        print('Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

    with torch.no_grad():
        correct = 0
        total = 0
        for batch in trainloader:
            images, labels = batch['image'].to(device), batch['label'].to(device)
            outputs, icnn_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        mlflow.log_metric("train_accuracy", 100 * correct / total)
        print('Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))
