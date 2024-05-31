import pandas as pd
import torch
from torch import nn
from tqdm.notebook import tqdm


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g., 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def get_current_lr(optimizer):
    """Retrieve the current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    """Train step for a single epoch.

    Args:
        model (torch.nn.Module): Model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device, optional): Device to use. Defaults to device.

    Returns:
        float: Average training loss for the epoch.
        float: Average training accuracy for the epoch.
    """
    train_loss, train_acc = 0, 0
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss += loss.item()

        y_pred = y_logits.argmax(dim=1)
        train_acc += accuracy_fn(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    """Validation step for a single epoch.

    Args:
        model (torch.nn.Module): Model to validate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_fn (torch.nn.Module): Loss function.
        device (torch.device, optional): Device to use. Defaults to device.

    Returns:
        float: Average validation loss for the epoch.
        float: Average validation accuracy for the epoch.
    """
    val_loss, val_acc = 0, 0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()

            y_pred = y_logits.argmax(dim=1)
            val_acc += accuracy_fn(y, y_pred)

    val_loss /= len(dataloader)
    val_acc /= len(dataloader)
    return val_loss, val_acc


def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    early_stopper,
    lr_scheduler,
    device: torch.device,
    return_df: bool = True,
    epochs: int = 5,
):
    """Train a model with early stopping and learning rate scheduling.

    Args:
        model (torch.nn.Module): Model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_dataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn (torch.nn.Module): Loss function.
        early_stopper: Early stopping callback.
        lr_scheduler: Learning rate scheduler.
        epochs (int, optional): Number of epochs to train. Defaults to 5.

    Returns:
        dict: Training and validation metrics.
        pd.DataFrame: DataFrame containing summary of results.

    Example usage:
        model = Model()
        train_dataloder = get_dataloader()
        val_dataloader = get_dataloader()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        early_stopper = EarlyStopping()
        lr_scheduler = lr_scheduler()
        epochs = 5
        results, df = train_model(model, train_dataloader, val_dataloader, optimizer, loss_fn, early_stopper, lr_scheduler, epochs)
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    early_stopped_epoch = epochs
    initial_lr = get_current_lr(optimizer)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = val_step(
            model=model, dataloader=val_dataloader, loss_fn=loss_fn, device=device
        )

        current_lr = get_current_lr(optimizer)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["lr"].append(current_lr)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"lr: {current_lr}"
        )

        lr_scheduler.step(val_loss)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            early_stopped_epoch = epoch + 1
            print(f"Early Stopping at Epoch: {epoch+1}")
            break

    early_stopper.load_best_weights(model)

    if return_df:
        df_results = {
            "Model": model.__class__.__name__,
            "Best Model Path": early_stopper.path,
            "Original Epochs": epochs,
            "Early Stopped Epoch": early_stopped_epoch,
            "Optimizer": optimizer.__class__.__name__,
            "Loss Function": loss_fn.__class__.__name__,
            "initial_lr": initial_lr,
            "final_lr": min(results["lr"]),
            "train_loss": min(results["train_loss"]),
            "train_acc": max(results["train_acc"]),
            "val_loss": min(results["val_loss"]),
            "val_acc": max(results["val_acc"]),
        }
        df = pd.DataFrame([df_results])
        return results, df
    else:
        return results
