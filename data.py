import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import albumentations as A
import torch
from torch.utils.data import Dataset
from torchvision import transforms


""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Dataset class for loading images and masks """
class CustomDataset(Dataset):
    def __init__(self, images, masks, augmentations=None):
        self.images = images
        self.masks = masks
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks[index]
        
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Change to grayscale

        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = np.transpose(image, (2, 0, 1))  # From HWC to CHW format for PyTorch
        mask = np.expand_dims(mask, axis=0)  # Add channel dimension to mask

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

def load_data(path, split=0.1):
    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    """ Split the data into training and testing """
    split_size = int(len(X) * split)
    
    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (test_x, test_y)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    augmentations_list = [
        A.HorizontalFlip(p=1.0),
        A.ChannelShuffle(p=1.0),
        A.CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32),
        A.Rotate(limit=45, p=1.0)
    ]

    for x, y in tqdm(zip(images, masks), total=len(images)):
        """ Extract the name """
        name = os.path.basename(x).split(".")[0]

        """ Reading the image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # Grayscale mask

        """ Augmentation """
        if augment:
            augmented_data = []

            for aug in augmentations_list:
                augmented = aug(image=x, mask=y)
                augmented_data.append((augmented['image'], augmented['mask']))

            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            augmented_data.append((x2, y))

            X = [x] + [a[0] for a in augmented_data]
            Y = [y] + [a[1] for a in augmented_data]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            try:
                """ Center Cropping """
                aug = A.CenterCrop(H, W, p=1.0)
                augmented = aug(image=i, mask=m)
                i = augmented["image"]
                m = augmented["mask"]

            except Exception as e:
                i = cv2.resize(i, (W, H))
                m = cv2.resize(m, (W, H))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the dataset """
    data_path = "people_segmentation"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Create directories to save the augmented data """
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """ Data augmentation """
    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)