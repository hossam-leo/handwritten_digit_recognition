import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def load_and_preprocess_data():
    """
    Loads MNIST dataset, normalizes pixel values, and splits into train/val/test sets for PyTorch.
    """
    print("Loading MNIST dataset...")
    
    # Define transformations: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Mean and std for MNIST
    ])

    # Download and load the training data
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training data into training and validation sets (90% train, 10% val)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    print(f"Data loaded successfully:")
    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset
