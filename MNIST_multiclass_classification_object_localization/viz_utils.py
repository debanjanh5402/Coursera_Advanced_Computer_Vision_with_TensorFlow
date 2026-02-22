import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from engine import compute_iou


def visualize_dataset_samples(dataset, num_samples=16, cols:int=4):
    """
    Visualizes a few samples from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to visualize.
        num_samples (int): Number of samples to visualize. Default is 16.
        cols (int): Number of columns in the visualization grid. Default is 4."""
    
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=300)
    axes = axes.flatten()
    
    for i in range(num_samples):
        img, label, bbox = dataset[i]
        h, w = img.shape[1], img.shape[2]
        xmin = int(bbox[0] * w)
        ymin = int(bbox[1] * h)
        xmax = int(bbox[2] * w)
        ymax = int(bbox[3] * h)
        axes[i].imshow(img.squeeze().numpy(), cmap='gray')
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 linewidth=1, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    # Hide any remaining subplots if num_samples is less than rows*cols
    for j in range(num_samples, rows*cols):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()



def visualize_image_with_bbox(image, bbox, title=None):
    """
    Visualizes an image with a bounding box.

    Args:
        image (torch.Tensor): Tensor of shape (1, H, W) containing the image.
        bbox (torch.Tensor): Tensor of shape (4,) containing the normalized bounding box coordinates [xmin, ymin, xmax, ymax].
        title (str, optional): Title for the plot. Default is None.
    """
    # Convert the image tensor to a numpy array and squeeze the channel dimension
    img = image.squeeze().numpy()
    
    # Get the height and width of the image
    h, w = img.shape
    
    # Denormalize the bounding box coordinates
    xmin = int(bbox[0] * w)
    ymin = int(bbox[1] * h)
    xmax = int(bbox[2] * w)
    ymax = int(bbox[3] * h)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1, figsize=(6, 6), dpi=300)
    
    # Display the image
    ax.imshow(img, cmap='gray')
    
    # Create a rectangle patch for the bounding box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                             linewidth=1, edgecolor='r', facecolor='none')
    
    # Add the rectangle patch to the axis
    ax.add_patch(rect)
    
    # Set title if provided
    if title:
        plt.title(title)

    ax.axis('off')  # Hide axes
    
    # Show the plot
    plt.show()



def visualize_predictions_on_dataset(dataset, model, num_samples:int=16, cols:int=4, device='cpu'):
    """
    Visualizes model predictions on a few samples from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to visualize.
        model (torch.nn.Module): The trained model for making predictions.
        num_samples (int): Number of samples to visualize. Default is 16.
        cols (int): Number of columns in the visualization grid. Default is 4.
        device (str): Device to run the model on. Default is 'cpu'.
    """
    model.eval()
    rows = math.ceil(num_samples / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=300)
    axes = axes.flatten()

    with torch.no_grad():
        for i in range(num_samples):
            img, label, bbox = dataset[i]
            img = img.to(device).unsqueeze(0)  # Add batch dimension
            pred_logits, pred_bbox = model(img)
            pred_label = torch.argmax(pred_logits, dim=1).item()
            pred_bbox = pred_bbox.squeeze().cpu()

            h, w = img.shape[2], img.shape[3]
            xmin_pred = int(pred_bbox[0] * w)
            ymin_pred = int(pred_bbox[1] * h)
            xmax_pred = int(pred_bbox[2] * w)
            ymax_pred = int(pred_bbox[3] * h)

            xmin_true = int(bbox[0] * w)
            ymin_true = int(bbox[1] * h)
            xmax_true = int(bbox[2] * w)
            ymax_true = int(bbox[3] * h)

            iou = compute_iou(pred_bbox.unsqueeze(0), bbox.unsqueeze(0), h=h, w=w).item()

            axes[i].imshow(img.squeeze().cpu().numpy(), cmap='gray')
            # Predicted bbox in red
            rect_pred = patches.Rectangle((xmin_pred, ymin_pred), xmax_pred - xmin_pred, 
                                          ymax_pred - ymin_pred, linewidth=1, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect_pred)
            # True bbox in green
            rect_true = patches.Rectangle((xmin_true, ymin_true), xmax_true - xmin_true, 
                                          ymax_true - ymin_true, linewidth=1, edgecolor='g', facecolor='none')
            axes[i].add_patch(rect_true)
            axes[i].set_title(f"True: {label} | Pred: {pred_label}, IOU: {iou:.2f}")
            axes[i].axis('off')

        # Hide any remaining subplots if num_samples is less than rows*cols
        for j in range(num_samples, rows*cols):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()