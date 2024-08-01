import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score

class TrajectoryModel(pl.LightningModule):
    def __init__(self):
        super(TrajectoryModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64*300, 128)
        self.fc2 = torch.nn.Linear(128, 5)

        self.accuracy = Accuracy(task='multiclass', num_classes=5)
        self.f1 = F1Score(task='multiclass', num_classes=5)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)

        acc = self.accuracy(preds, labels)
        f1 = self.f1(preds, labels)
        self.log('training_loss', loss)
        self.log('training_accuracy', acc)
        self.log('training_f1', f1)

        return loss

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim = 1)

        acc = self.accuracy(preds, labels)
        f1 = self.f1(preds, labels)
        self.log('validation_loss', loss, prog_bar = True)
        self.log('validation_accuracy', acc, prog_bar = True)
        self.log('val_f1', f1, prog_bar = True) 

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
