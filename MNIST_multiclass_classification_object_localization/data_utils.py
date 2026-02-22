import torch
from torch.utils.data import dataset

from typing import Tuple

class SytheticMNISTDataset(dataset.Dataset):
    def __init__(self, 
                 data:torch.Tensor, 
                 labels:torch.Tensor,
                 canvas_size:Tuple[int, int]=(128, 128)):
        """
        Args:
            data (torch.Tensor): Tensor of shape (N, H, W) containing the images.
            labels (torch.Tensor): Tensor of shape (N, ) containing the labels.
            canvas_size (Tuple[int, int]): Size of the canvas on which the digits are placed. Default is (128, 128).
        """
        
        self.data = data
        self.labels = labels
        self.canvas_h, self.canvas_w = canvas_size
        self.digit_h, self.digit_w = data.shape[1], data.shape[2]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = self.data[idx]
        label = self.labels[idx]
        canvas = torch.zeros(size=(1, self.canvas_h, self.canvas_w), dtype=torch.float32)
        ymin = torch.randint(0, self.canvas_h - self.digit_h, (1,)).item()
        xmin = torch.randint(0, self.canvas_w - self.digit_w, (1,)).item()
        ymax = ymin + self.digit_h
        xmax = xmin + self.digit_w
        canvas[:, ymin:ymax, xmin:xmax] = image
        bbox = torch.tensor([xmin/self.canvas_w, ymin/self.canvas_h, 
                             xmax/self.canvas_w, ymax/self.canvas_h], dtype=torch.float32)

        return canvas, label, bbox