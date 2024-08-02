import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet1d import BasicBlock, ResNet


class BackboneNet(nn.Module):
    def __init__(
        self,
        backbone,
        num_classes=None,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels
        resolution = 8
        representation_size = 32

        predictor_blocks = [nn.AdaptiveMaxPool1d(output_size=resolution)]
        prev_channels = out_channels
        for channels in [256, 128, 64]:
            predictor_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=prev_channels, out_channels=channels, kernel_size=7, padding=3), 
                    nn.BatchNorm1d(channels), 
                    nn.ReLU(inplace=True)
                )
            )
            prev_channels = channels
        predictor_blocks.append(nn.Sequential(
            nn.Flatten(), 
            nn.Linear(prev_channels * resolution, representation_size), 
            nn.ReLU(inplace=True), 
            nn.Linear(representation_size, num_classes)
        ))
        self.predictor = nn.Sequential(*predictor_blocks)
        
        for layer in self.predictor.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, Xs):
        features = self.backbone(Xs) # 5 of 256 x (125, 63, 32, 16, 8)
        scores = []
        for item in list(features):
            scores.append(self.predictor(item)) # 5 of 8
        return scores

class BackboneNetWrap:
    def __init__(self, learning_rate=1e-3, num_classes=9, device='cuda') -> None:
        super().__init__()
        self.device = device
        self.model = BackboneNet(backbone=ResNet(BasicBlock, [2, 2, 2, 2]), num_classes=num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)

    def train_step(self, train_loader, verbose=False, file=None):
        self.model.train()
        train_loss = np.zeros(5)
        for batch, data in enumerate(train_loader):
            X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
            preds = self.model(X)
            
            loss = 0
            for pred in preds:
                loss += self.loss_fn(pred, y)
            if verbose:
                if file is None: 
                    print('{} / {}, loss: {}'.format(batch, len(train_loader), loss))
                else:
                    print('{} / {}, loss: {}'.format(batch, len(train_loader), loss), file=file)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += np.array([(
                torch.argmax(preds[i], dim=1) != y).float().sum().cpu().item() for i in range(5)])
        return train_loss

    def test(self, test_loader, epoch=1):
        self.model.eval()
        self.target = []
        self.prediction = [[], [], [], [], []]
        for i1 in range(epoch):
            for batch, data in enumerate(test_loader):
                X, y = data['data'].to(device=self.device), data['label'].to(device=self.device)
                scores = self.model(X)
                
                for i2, score in enumerate(scores):
                    _, pred = torch.max(score.data, 1)
                    self.prediction[i2] += pred.cpu().tolist()
                self.target += y.cpu().tolist()
        return self.target, self.prediction