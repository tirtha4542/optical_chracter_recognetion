# %%
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
from PIL import Image
import pytorch_lightning as pl
from glob import glob
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
# %%
ALL_CHAR = '0123456789abcdefghijklmnopqrstuvwxyz'
LEN_ALL_CHAR = len(ALL_CHAR)
MAX_CAPTCHA = 4
class OCRDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def encode(self, char):
        onehot = [0] * LEN_ALL_CHAR
        idx = ALL_CHAR.index(char)
        onehot[idx] += 1
        return onehot

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img = Image.open(image_file).convert("L")
        label = image_file.split(os.sep)[-1][:-4]
        label_oh = []
        for i in label[:MAX_CAPTCHA]:
            label_oh += self.encode(i)
        label_oh += [0] * (LEN_ALL_CHAR * (MAX_CAPTCHA - len(label)))
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label

class OCRDATAMODULE(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize([64, 128]),
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        train_image_files = glob(r"E:\capstone_project_1\project4\data\OCR\Train\*")
        test_image_file = glob(r"E:\capstone_project_1\project4\data\OCR\Test\*")
        self.train_dataset = OCRDataset(train_image_files, transform=self.transform)
        self.val_dataset = OCRDataset(test_image_file, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):  # test_step এর জন্য প্রয়োজন
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)



class OCRModel(pl.LightningModule):
    def __init__(self, num_char=LEN_ALL_CHAR, max_capcha=MAX_CAPTCHA):
        super(OCRModel, self).__init__()
        self.save_hyperparameters()
        self.char_set = ALL_CHAR

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_char * max_capcha)
        )

        self.train_acc_metric = Accuracy(task='multiclass', num_classes=num_char)
        self.val_acc_metric = Accuracy(task='multiclass', num_classes=num_char)

        self.inference_transform = transforms.Compose([
            transforms.Resize([64, 128]),
            transforms.ToTensor(),
        ])

    def forward(self, x):
        out = self.resnet(x)
        out = out.view(-1, self.hparams.max_capcha, self.hparams.num_char)
        return out

    def process_batch(self, batch):
        x, y, _ = batch
        y_hat = self(x)
        batch_size = x.size(0)
        y_hat_flat = y_hat.view(batch_size * self.hparams.max_capcha, self.hparams.num_char)
        y_flat = y.view(batch_size, self.hparams.max_capcha, self.hparams.num_char).argmax(dim=-1).view(-1)
        return y_hat_flat, y_flat, batch_size

    def training_step(self, batch, batch_idx):
        y_hat, y, b_size = self.process_batch(batch)
        loss = F.cross_entropy(y_hat, y)
        acc = self.train_acc_metric(y_hat.softmax(dim=-1), y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=b_size)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, batch_size=b_size)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y, b_size = self.process_batch(batch)
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_acc_metric(y_hat.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=b_size)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=b_size)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y, b_size = self.process_batch(batch)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, batch_size=b_size)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def predict_image(self, image_path):
        img = Image.open(image_path).convert('L')
        img = self.inference_transform(img).unsqueeze(0).to(self.device)
        self.eval()
        with torch.no_grad():
            output = self(img).softmax(dim=-1)
            pred_indices = output.argmax(dim=-1)
        return ''.join(self.char_set[idx] for idx in pred_indices[0])


mlf_logger = MLFlowLogger(
    experiment_name="Captcha_OCR_Project",
    tracking_uri="file:./mlruns"
)

data_module = OCRDATAMODULE(batch_size=32)
model = OCRModel()

trainer = pl.Trainer(
    max_epochs=50,
    logger=mlf_logger,
    log_every_n_steps=1
)


trainer.fit(model, data_module)

image_path = r"E:\capstone_project_1\project4\data\OCR\Test\4htx.png"
if os.path.exists(image_path):
    pred_text = model.predict_image(image_path)
    print(f"Predicted Text: {pred_text}")
else:
    print("Image path not found!")


print("\n" + "=" * 70)
print(f'mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}')
print("=" * 70 + "\n")