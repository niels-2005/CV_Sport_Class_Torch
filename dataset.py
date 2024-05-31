from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class ImageDataset(Dataset):
    """Custom Dataset for loading images from a directory.

    Args:
        data_dir (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        subset (str, optional): Subset to use ('train', 'val', or 'all'). Defaults to 'all'.
        val_split (float, optional): Fraction of the data to use for validation. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Attributes:
        data (ImageFolder): Instance of ImageFolder with the provided directory and transform.
        filepaths (list): List of file paths for each image in the dataset.
        labels (list): List of labels for each image in the dataset.
        indices (list): List of indices for each image in the dataset.
        train_indices (list): List of indices for the training subset.
        val_indices (list): List of indices for the validation subset.
    """

    def __init__(self, data_dir, transform=None, subset='all', val_split=0.2, seed=42):
        self.data = ImageFolder(data_dir, transform=transform)
        self.filepaths = [s[0] for s in self.data.samples]
        self.labels = [s[1] for s in self.data.samples]
        self.indices = list(range(len(self.data)))
        self.train_indices, self.val_indices = train_test_split(
            self.indices, test_size=val_split, random_state=seed, stratify=self.labels)

        if subset == 'train':
            self.indices = self.train_indices
        elif subset == 'val':
            self.indices = self.val_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return self.data[index]

    @property
    def classes(self):
        return self.data.classes

    def get_filepaths(self):
        return [self.filepaths[i] for i in self.indices]

    def get_labels(self):
        return [self.labels[i] for i in self.indices]

    def get_indices(self):
        return self.indices

    def get_train_indices(self):
        return self.train_indices

    def get_val_indices(self):
        return self.val_indices


def get_datasets(train_folder, val_folder, test_folder):
    """Creates datasets for training, validation, and testing.

    Args:
        train_folder (str): Path to the training data directory.
        val_folder (str): Path to the validation data directory.
        test_folder (str): Path to the testing data directory.

    Returns:
        tuple: A tuple containing the training, validation, and testing datasets.

    Example usage:
        train_dataset, val_dataset, test_dataset = get_datasets(
            train_folder="path/to/train",
            val_folder="path/to/val",
            test_folder="path/to/test"
        )
    """
    # Train Augmentations if needed
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # Datasets
    train_dataset = ImageDataset(data_dir=train_folder, transform=train_transform)
    val_dataset = ImageDataset(data_dir=val_folder, transform=test_transform)
    test_dataset = ImageDataset(data_dir=test_folder, transform=test_transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, batch_size):
    """Creates dataloaders for training, validation, and testing datasets.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        test_dataset (Dataset): Testing dataset.
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple: A tuple containing the training, validation, and testing dataloaders.

    Example usage:
        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            train_dataset, val_dataset, test_dataset, batch_size=32
        )
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader