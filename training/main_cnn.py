import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from PIL import Image
from torchvision.transforms import (Compose, Normalize, Resize, ToPILImage,
                                    ToTensor)
from tqdm import tqdm, trange
import numpy as np
import logging
from matplotlib.cm import ScalarMappable
from torchvision import transforms
import torchvision
from matplotlib.colors import Normalize
import random
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from training.model import Model
from training.model_icnn import Model as ModelICNN
import mlflow
import mlflow.pytorch
import time
import torchvision.transforms.functional as F
from training.loss import mutual_information

mlflow.set_experiment("non-interpretable-cnn")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

DESIRED_SHAPE = 64
transform = transforms.Compose(
    [  
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.5), (0.5)),
        transforms.Resize((DESIRED_SHAPE, DESIRED_SHAPE)),
        transforms.CenterCrop((DESIRED_SHAPE, DESIRED_SHAPE)),
        # transforms.ColorJitter(brightness=1.8),
    ]    
    )

batch_size = 24

dataset = load_dataset("Hemg/Melanoma-Cancer-Image-Dataset", split="train")
trainset, testset = dataset.train_test_split(test_size=0.1, seed=42).values()

trainset = trainset.with_format("torch")
testset  = testset.with_format("torch")

trainset = trainset.with_transform(lambda examples: {'image': [transform(img) for img in examples['image']], 'label': examples['label']})

testset = testset.with_transform(lambda examples: {'image': [transform(img) for img in examples['image']], 'label': examples['label']})

classes = trainset.features['label'].names
classes

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)


INPUT_CHANNELS = 3

n = trainset[0]['image'].shape[-1]
print("Feature map size n =", n)


lambda_ = 1e-2
EPOCHS = 20

losses = {
    "total_loss": [],
    "class_loss": [],
    "interp_loss": []
}

model = Model(feature_map_size=n, channels=3, num_classes=len(classes), filters_icnn=12)
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

current_date = time.strftime("%Y%m%d-%H%M%S")

checkpoint_every = 5
checkpoint_path = f"model_cnn_checkpoint_{current_date}.pth"

with mlflow.start_run(run_name=f"run_{current_date}"):
    mlflow.log_param("lambda", lambda_)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("feature_map_size", n)

    for epoch in trange(EPOCHS):
        losses_epoch = {
            "total_loss": [],
            "class_loss": []
        }

        for batch in tqdm(trainloader, desc="Training", leave=False):

            optimizer.zero_grad()
            images, labels = batch['image'].to(device), batch['label'].to(device)
            logits = model(images)

            class_loss = criterion(logits, labels.long())


            losses_epoch["total_loss"].append(class_loss.item())
            losses_epoch["class_loss"].append(class_loss.item())

            class_loss.backward()
            optimizer.step()
        for key in losses_epoch:
            mlflow.log_metric(f"train_{key}", np.mean(losses_epoch[key]), step=epoch)
        
        if (epoch + 1) % checkpoint_every == 0:
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_artifact(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
      

torch.save(model.state_dict(), f"model_cnn_{current_date}.pth")

mlflow.pytorch.log_model(model, artifact_path=f"model_cnn_{current_date}", registered_model_name="cnn-model")

# Evaluate
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in testloader:
        images, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted)
        # print(labels)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))