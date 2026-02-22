import torch
from torch import nn


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, apply_norm=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if apply_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def feature_extractor(in_channels=1):
    return nn.Sequential(
        conv_block(in_channels, 16, apply_norm=False), # (bs, 16, 64, 64)
        conv_block(16, 32), # (bs, 32, 32, 32)
        conv_block(32, 64), # (bs, 64, 16, 16),
        conv_block(64, 128) # (bs, 128, 8, 8)
)


def dense_block(in_features, out_features, apply_norm=True):
    layers = [nn.Linear(in_features, out_features)]
    if apply_norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class ObjectLocalizationModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.feature_extractor = feature_extractor(1)
        self.dense_layer = dense_block(128 * 8 * 8, 128)

        self.classifier = nn.Linear(128, num_classes)
        self.bbox_regressor = nn.Sequential(nn.Linear(128, 4),
                                            nn.Sigmoid())

    def forward(self, x):
        features = self.feature_extractor(x)
        features_flat = torch.flatten(features, start_dim=1)
        dense_output = self.dense_layer(features_flat)

        class_logits = self.classifier(dense_output)
        bbox_coords = self.bbox_regressor(dense_output)

        return class_logits, bbox_coords