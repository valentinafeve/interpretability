import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, ToPILImage

class CatDogDataset(Dataset):
    def __init__(self, annotations_file, dataset_path):

        transform = Compose([
            ToPILImage(),
            Resize((128, 128)),
            ToTensor(),
            Normalize((0.5,), (0.5,))
        ])
        self.transform = transform

        cat_files = sorted([c for a, b, c in os.walk(os.path.join(dataset_path, 'Cat'))][0])
        dog_files = sorted([c for a, b, c in os.walk(os.path.join(dataset_path, 'Dog'))][0])

        self.files = pd.concat([
            pd.DataFrame({'filename': cat_files, 'label': 0}),
            pd.DataFrame({'filename': dog_files, 'label': 1})
        ], ignore_index=True)
        self.files["img_path"] = self.files[["filename", "label"]].apply(
            lambda x: os.path.join(dataset_path, "Dog" if x["label"] else "Cat", x["filename"]), axis=1
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files.iloc[idx]
        img_path = os.path.join(item["img_path"])
        image = read_image(img_path)

        label = item["label"]
        if self.transform:
            image = self.transform(image)
        return image, label