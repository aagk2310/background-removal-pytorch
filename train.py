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
from model import DeepLabV3Plus  # Assuming your converted PyTorch DeepLabV3Plus model is here
from metrics import dice_loss, dice_coef, iou  # You need to define these for PyTorch

# Login to WandB (it will prompt for your API key if you're not logged in)
wandb.login()

""" Global parameters """
H = 512
W = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*png")))
    y = sorted(glob(os.path.join(path, "mask", "*png")))
    return x, y

def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)  # Adding channel dimension
    return x

class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_image(self.images[index])
        mask = read_mask(self.masks[index])

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return torch.tensor(image.transpose(2, 0, 1)), torch.tensor(mask)

def get_loader(images, masks, batch_size, shuffle_data=True):
    dataset = CustomDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data, num_workers=4)

# Function to calculate accuracy (Dice coefficient in this case)
def calculate_accuracy(outputs, masks):
    outputs = torch.sigmoid(outputs)  # Convert logits to probabilities
    outputs = (outputs > 0.5).float()  # Apply threshold
    intersection = torch.sum(outputs * masks)
    union = torch.sum(outputs) + torch.sum(masks)
    dice = (2.0 * intersection + 1e-15) / (union + 1e-15)
    return dice.item()

def train_epoch(model, loader, criterion, optimizer, device, epoch, print_every=100):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batch_count = 0

    batch_loss_acc = 0.0
    batch_acc_acc = 0.0
    for i, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(outputs, masks)
        running_loss += loss.item()
        running_acc += acc

        # Accumulate loss and accuracy for printing
        batch_loss_acc += loss.item()
        batch_acc_acc += acc
        batch_count += 1

        # Print the mean loss and accuracy every 'print_every' batches
        if (i + 1) % print_every == 0:
            avg_loss = batch_loss_acc / print_every
            avg_acc = batch_acc_acc / print_every
            print(f"Train Epoch {epoch+1}, Batch {i+1}/{len(loader)} (last {print_every} batches): "
                  f"Mean Loss = {avg_loss:.4f}, Mean Accuracy = {avg_acc:.4f}")
            batch_loss_acc = 0.0  # Reset for next batch group
            batch_acc_acc = 0.0

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc

def validate_epoch(model, loader, criterion, device, epoch, print_every=100):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    batch_count = 0

    batch_loss_acc = 0.0
    batch_acc_acc = 0.0
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            acc = calculate_accuracy(outputs, masks)
            running_loss += loss.item()
            running_acc += acc

            # Accumulate loss and accuracy for printing
            batch_loss_acc += loss.item()
            batch_acc_acc += acc
            batch_count += 1

            # Print the mean loss and accuracy every 'print_every' batches
            if (i + 1) % print_every == 0:
                avg_loss = batch_loss_acc / print_every
                avg_acc = batch_acc_acc / print_every
                print(f"Val Epoch {epoch+1}, Batch {i+1}/{len(loader)} (last {print_every} batches): "
                      f"Mean Loss = {avg_loss:.4f}, Mean Accuracy = {avg_acc:.4f}")
                batch_loss_acc = 0.0  # Reset for next batch group
                batch_acc_acc = 0.0

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)
    return avg_loss, avg_acc

if __name__ == "__main__":
    """ Initialize WandB """
    wandb.init(project="DeepLabV3Plus-Project")

    """ Seeding """
    np.random.seed(42)
    torch.manual_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 8
    lr = 1e-4
    num_epochs = 8
    model_path = os.path.join("files", "model.pth")
    log_dir = os.path.join("logs")

    """ Dataset """
    dataset_path = "new_data"
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "test")

    train_x, train_y = load_data(train_path)
    train_x, train_y = shuffling(train_x, train_y)
    valid_x, valid_y = load_data(valid_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")

    train_loader = get_loader(train_x, train_y, batch_size=batch_size)
    valid_loader = get_loader(valid_x, valid_y, batch_size=batch_size, shuffle_data=False)

    """ Model, Loss, Optimizer """
    model = DeepLabV3Plus(num_classes=1).to(device)
    model.load_state_dict(torch.load('/notebooks/files/model.pth'))
    criterion = dice_loss  # You need to implement dice_loss for PyTorch
    optimizer = optim.Adam(model.parameters(), lr=lr)

    """ Log Hyperparameters to WandB """
    wandb.config.update({
        "learning_rate": lr,
        "epochs": num_epochs,
        "batch_size": batch_size
    })

    best_loss = float("inf")
    patience = 5
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Train and validate for the current epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = validate_epoch(model, valid_loader, criterion, device, epoch)

        # Print epoch-level losses and accuracies
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        """ Log metrics to WandB """
        wandb.log({
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Val Loss": val_loss,
            "Val Accuracy": val_acc,
            "epoch": epoch
        })

        """ Model Checkpointing """
        if val_loss < best_loss:
            print(f"Validation loss decreased from {best_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)  # Save model to WandB
            best_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping!")
                break

    wandb.finish()