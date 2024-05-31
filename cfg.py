import os

import torch

from callbacks import EarlyStopping
from dataset import get_dataloader, get_datasets
from model import HotdogClassifier


class CFG:
    """
    Configuration class for setting up training and evaluation parameters.

    Attributes:
        folder (str): Path to the main dataset folder.
        train_folder (str): Path to the training dataset folder.
        val_folder (str): Path to the validation dataset folder.
        test_folder (str): Path to the test dataset folder.
        batch_size (int): Batch size for data loaders.
        epochs (int): Number of epochs to train the model.
        plot_image_at_begin (bool): Flag to plot images at the beginning of training.
        model (torch.nn.Module): The model to be trained.
        model_name (str): Name of the model.
        model_folder (str): Folder to save the model checkpoints.
        save_model_path (str): Path to save the trained model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_fn (torch.nn.Module): Loss function for training.
        early_stopper (EarlyStopping): Early stopping mechanism to prevent overfitting.
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        train_dataset (torch.utils.data.Dataset): Training dataset.
        val_dataset (torch.utils.data.Dataset): Validation dataset.
        test_dataset (torch.utils.data.Dataset): Test dataset.
        class_names (list): List of class names in the dataset.
        train_dataloader (torch.utils.data.DataLoader): Data loader for training dataset.
        val_dataloader (torch.utils.data.DataLoader): Data loader for validation dataset.
        test_dataloader (torch.utils.data.DataLoader): Data loader for test dataset.
        device (str): Device to run the training (cuda or cpu).
    """

    # folder = "./hotdog-nothotdog"
    train_folder = "./train"
    val_folder = "./test"
    test_folder = "./test"

    batch_size = 32
    epochs = 3

    plot_image_at_begin = True

    model = HotdogClassifier()
    model_name = model.__class__.__name__
    model_folder = model.__class__.__name__ + "/"
    save_model_path = model_folder + "model.pt"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=10, path=save_model_path, verbose=True)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.1, patience=3
    )

    train_dataset, val_dataset, test_dataset = get_datasets(
        train_folder, val_folder, test_folder
    )
    class_names = train_dataset.classes

    train_dataloader, val_dataloader, test_dataloader = get_dataloader(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
