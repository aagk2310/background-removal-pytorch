import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.utils import shuffle
import wandb
from model import DeepLabV3Plus 
from metrics import dice_loss, dice_coef, iou 
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as transforms


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepLabV3Plus(num_classes=1).to(device)
    model.load_state_dict(torch.load('files/model.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for inference
        img = read_image('test-10.jpg').to(device)  # Ensure image is on the same device
        img_temp = img.clone()  # Clone the image for later visualization
        h, w = img_temp.shape[1], img_temp.shape[2]  # Get original image height and width

        img = img.float() / 255.0  # Normalize to [0, 1]
        img = transforms.Resize((512, 512))(img)  # Resize for model input
        img = img.unsqueeze(0)  # Add a batch dimension

        # Forward pass through the model to get mask prediction
        mask_predicted = model(img)
        mask_predicted = transforms.Resize((h, w))(mask_predicted)
        mask_predicted = (mask_predicted >= 0.5).float()  # Apply threshold to get binary mask

        # Ensure mask has same number of channels as the image (3 channels)
        mask_predicted = mask_predicted.repeat(1, 3, 1, 1)  # Repeat across 3 channels
         # Resize mask to original dimensions
        print(mask_predicted.unique())
        mask_predicted=mask_predicted.squeeze()
        print(mask_predicted.shape)
        # Remove background: element-wise multiplication of mask with the original image
        img_bg_removed = mask_predicted * img_temp

        # Remove batch dimension and move to CPU for visualization
        img_bg_removed = img_bg_removed.int().cpu()
       
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(img_bg_removed.permute(1, 2, 0))  # Display background removed image
        ax[0].axis('off')  
        ax[0].set_title('Background Removed')

        ax[1].imshow(img_temp.cpu().permute(1, 2, 0))  # Display original image
        ax[1].axis('off')  
        ax[1].set_title('Original Image')

        plt.tight_layout()
        plt.show()

