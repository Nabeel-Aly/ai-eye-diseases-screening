import os
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EyeDiseaseDataset:
    def __init__(
        self,
        data_dir: str,
        img_size: int = 224,
        batch_size: int = 32,
        val_split: float = 0.2,
        num_workers: int = 2
    ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

    def get_dataloaders(self):
        full_dataset = ImageFolder(
            root=self.data_dir,
            transform=self._apply_train_transform
        )

        val_size = int(len(full_dataset) * self.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )

        # Override validation transform
        val_dataset.dataset.transform = self._apply_val_transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

        return train_loader, val_loader, full_dataset.classes

    def _apply_train_transform(self, image):
        image = self._to_numpy(image)
        return self.train_transform(image=image)["image"]

    def _apply_val_transform(self, image):
        image = self._to_numpy(image)
        return self.val_transform(image=image)["image"]

    @staticmethod
    def _to_numpy(image):
        import numpy as np
        return np.array(image)
