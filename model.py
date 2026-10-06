import torch
import torch.nn as nn


class DisasterNet(nn.Module):
    """Multi-branch network: CNN (satellite) + Dense (IoT) + LSTM (weather) -> fusion -> 2 heads."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 4 * 4, 32), nn.ReLU())
        self.sensor_branch = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.weather_branch = nn.LSTM(4, 24, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(32 + 16 + 24, 64), nn.ReLU(), nn.Dropout(0.1))
        self.hazard_head = nn.Linear(64, n_classes)
        self.risk_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, img, sensor, weather):
        a = self.image_branch(img)
        b = self.sensor_branch(sensor)
        _, (h, _) = self.weather_branch(weather)
        z = self.fusion(torch.cat([a, b, h[-1]], dim=1))
        return self.hazard_head(z), self.risk_head(z).squeeze(1)
