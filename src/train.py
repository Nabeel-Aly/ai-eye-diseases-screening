from data.dataloader import EyeDiseaseDataset

dataset = EyeDiseaseDataset(
    data_dir="dataset",
    img_size=224,
    batch_size=32
)

train_loader, val_loader, class_names = dataset.get_dataloaders()

print("Classes:", class_names)
