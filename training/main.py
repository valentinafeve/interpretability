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

from training.loss import mutual_information
from training.model import Model
# from training.model_icnn import Model
from training.utils.plots import plot_tensors
from training.icnn import generate_templates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IS_INTERPRETABLE = True
## Hyperparameters for training
BATCH_SIZE = 24
EPOCHS = 30
DESIRED_IMAGE_SHAPE = 64
LAMBDA_ = 1e-3

logger.info(f"Using interpretable model: {IS_INTERPRETABLE}")
logger.info(f"Starting training with hyperparameters:" f"lambda={LAMBDA_}, epochs={EPOCHS}, image_shape={DESIRED_IMAGE_SHAPE}")

if IS_INTERPRETABLE:
    EXPERIMENT_NAME = "interpretable-cnn"
else:
    EXPERIMENT_NAME = "cnn"

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

model = Model(feature_map_size=n, channels=3, num_classes=len(classes), filters_icnn=12)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

current_date = time.strftime("%Y%m%d-%H%M%S")

checkpoint_every = 5
checkpoint_path = f"model_icnn_checkpoint_{current_date}.pth"

with mlflow.start_run(run_name=f"run_{current_date}"):
    mlflow.log_param("lambda", LAMBDA_)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("feature_map_size", n)

    # templates = generate_templates(32, tau=0.5, beta=4.0)
    for epoch in trange(EPOCHS):
        losses_epoch = {
            "total_loss": [],
            "class_loss": [],
            "interp_loss": []
        }

        templates = generate_templates(32, tau=0.5, beta=4.0).to(device)
        model.train()
        for batch in tqdm(trainloader, desc="Training", leave=False):
            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits, icnn_output = model(images)

            class_loss = criterion(logits, labels.long())
       
            if IS_INTERPRETABLE:
                interp_loss = mutual_information(icnn_output, templates, icnn_output.shape[-1], device=device).sum()
                total_loss = LAMBDA_ * interp_loss + (1 - LAMBDA_) * class_loss
                # total_loss = class_loss
                # losses_epoch["interp_loss"].append(interp_loss.item())

            else:
                total_loss = class_loss

            losses_epoch["total_loss"].append(total_loss.item())
            losses_epoch["class_loss"].append(class_loss.item())
    
            total_loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in valloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs, icnn_ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
            logger.info(f"Accuracy of the model on the {total} validation images: {100 * correct / total} %")
            mlflow.log_metric("val_accuracy", 100 * correct / total, step=epoch)

            correct = 0
            total = 0
            for batch in trainloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs, icnn_ = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            logger.info(f"Accuracy of the model on the {total} train images: {100 * correct / total} %")
            mlflow.log_metric("train_accuracy", 100 * correct / total, step=epoch)

        for key in losses_epoch:
            mlflow.log_metric(f"train_{key}", np.mean(losses_epoch[key]), step=epoch)
        if (epoch + 1) % checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path)
            logger.info(f"Checkpoint saved at epoch {epoch+1}")
            

    torch.save(model.state_dict(), f"model_icnn_{current_date}.pth")

    mlflow.pytorch.log_model(model, artifact_path=f"model_icnn_{current_date}", registered_model_name="interpretable-cnn-model")

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

    # Plot filters
    random_i = random.randint(0, len(testset) - 1)
    image, label = testset[random_i]['image'], testset[random_i]['label']
    image = image.unsqueeze(0).to(device)

    outputs, icnn_output = model(image)

    save_path = f"icnn_filters_{current_date}.png"
    img = plot_tensors(icnn_output, save_path=save_path)

    mlflow.log_image(img, "icnn_filters.png")