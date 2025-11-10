
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForImageClassification
from tqdm import tqdm

class CustomSegformerDataset(Dataset):
    def __init__(self, root_dir, feature_extractor,mode="train"):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.images = []
        self.labels = []
        self.mode=mode
        datafolder= os.path.join(self.root_dir, self.mode)

        if not os.path.isdir(datafolder):
            raise FileNotFoundError(f"Data Folder not found at: {datafolder}")
        # paths & labels
        for label_name in sorted(os.listdir(datafolder)):
            class_path = os.path.join(datafolder, label_name)
            if not os.path.isdir(class_path):
                continue
            try:
                label_name= int(label_name)
            except ValueError:
                continue
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(label_name)
        assert len(self.images) == len(self.labels), "Number of images and labels must match"
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # (3,H,W)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


train=CustomSegformerDataset("D:/Inteview/Dataset", feature_extractor=SegformerImageProcessor(do_resize=True))
#print(train[0])
print(shape:=train[0]['pixel_values'].shape)  # Check the shape of pixel values
print(f"Number of training samples: {len(train)}")
print(train[0]["labels"].shape)
print(train[0]["labels"])