import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import albumentations as A 
from dataset import CardiacDataset

# Data paths
train_root_path = "Processed-Heart-Detection/train/"
train_subjects = "train_subjects.npy"
val_root_path = "Processed-Heart-Detection/val/"
val_subjects = "val_subjects.npy"

# Simple training transformations
train_transforms = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
    A.Affine(scale=(0.8, 1.2), rotate=(-10, 10), p=0.9),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Create datasets
train_dataset = CardiacDataset("./rsna_heart_detection.csv", train_subjects, train_root_path, train_transforms)
val_dataset = CardiacDataset("./rsna_heart_detection.csv", val_subjects, val_root_path, None)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)} images")

# Create data loaders
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Simple model
class CardiacDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Use ResNet18 as backbone
        self.model = torchvision.models.resnet18(pretrained=True)
        # Change input: 3 channels -> 1 channel (grayscale)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Change output: 1000 classes -> 4 coordinates
        self.model.fc = torch.nn.Linear(512, 4)
        # Loss function for regression
        self.loss_fn = torch.nn.MSELoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        pred = self(x_ray)
        loss = self.loss_fn(pred, label.float())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        pred = self(x_ray)
        loss = self.loss_fn(pred, label.float())
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Create model and trainer
model = CardiacDetectionModel()

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    filename='{epoch:02d}-{val_loss:.3f}'
)

trainer = pl.Trainer(
    accelerator='cpu',
    max_epochs=100,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger("./logs"),
    log_every_n_steps=50
)

# Train the model
if __name__ == "__main__":
    trainer.fit(model, train_loader, val_loader)