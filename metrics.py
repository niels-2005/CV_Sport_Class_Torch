import torch
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics import ConfusionMatrix
from tqdm.notebook import tqdm
import os 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import classification_report


def get_y_pred_multilabel(model, test_loader, device):
    """Generates predictions for a multilabel classification model on a given test dataset.

    Args:
        model (torch.nn.Module): A PyTorch model for multilabel classification.
        test_loader (torch.utils.data.DataLoader): DataLoader containing the test dataset.

    Returns:
        torch.Tensor: A tensor containing the predicted labels for the test dataset.

    Example usage:
        model = YourMultilabelModel()
        test_loader = get_dataloader()
        y_pred = get_y_pred_multilabel(model=model, test_loader=test_loader)
    """
    y_preds = []
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(test_loader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            # Do the forward pass
            y_logit = model(X)
            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(
                dim=1
            )  # Note: perform softmax on the "logits" dimension, not "batch" dimension
            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())
    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)
    return y_pred_tensor


def get_y_true(dataset):
    """Extracts true labels from a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset containing the true labels.

    Returns:
        torch.Tensor: A tensor containing the true labels.

    Example usage:
        test_dataset = ImageDataset()
        y_true = get_y_true(dataset=test_dataset)
    """
    y_true = [label for _, label in dataset]
    y_true_tensor = torch.tensor(y_true)
    return y_true_tensor


def plot_metrics(model, dataloader, dataset, class_names, device, model_folder):
    """Plots a confusion matrix for a given binary classification model and dataset.

    Args:
        class_names (list): List of class names for the binary classification task.
        model (torch.nn.Module): A PyTorch model to generate predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        dataset (torch.utils.data.Dataset): The test dataset containing the true labels.
        device (torch.device): The device to run the model on.

    Example usage:
        classes = ["Class 0", "Class 1"]
        model = YourBinaryModel()
        dataloader = get_dataloader()
        dataset = ImageDataset()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        plot_metrics(model=model, dataloader=dataloader, dataset=dataset, class_names=classes, device=device)
    """
    y_pred = get_y_pred_multilabel(model=model, test_loader=dataloader, device=device)
    y_true = get_y_true(dataset=dataset)

    get_classes_metrics(
        y_true=y_true, y_pred=y_pred, class_names=class_names, model_folder=model_folder
    )

    confmat = ConfusionMatrix(num_classes=len(class_names), task="multiclass")
    confmat_tensor = confmat(preds=y_pred, target=y_true)

    # Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=class_names,  # Turn the row and column labels into class names
        figsize=(10, 7),
    )

    plt.title(f"Confusion Matrix: {model.__class__.__name__}")
    plot_path = os.path.join(model_folder, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

    return y_pred, y_true


def get_classes_metrics(y_true, y_pred, class_names, model_folder):
    """
    Generates a classification report for each class and saves it as a CSV file.

    Args:
        y_true (torch.Tensor or list): True labels.
        y_pred (torch.Tensor or list): Predicted labels.
        class_names (list): List of class names.

    Example usage:
        y_true = [0, 1, 0, 1, 1]
        y_pred = [0, 0, 0, 1, 1]
        class_names = ["Class 0", "Class 1"]
        model_folder = "path/to/save"

        get_classes_metrics(y_true=y_true, y_pred=y_pred, class_names=class_names, model_folder=model_folder)
    """
    report = classification_report(
        y_pred=y_pred, y_true=y_true, target_names=class_names, output_dict=True
    )

    report.pop("accuracy", None)
    report.pop("macro avg", None)
    report.pop("weighted avg", None)

    # generate pandas dataframe
    df = pd.DataFrame.from_dict(report).transpose().reset_index()
    df.rename(columns={"index": "class_name"}, inplace=True)

    csv_path = os.path.join(model_folder, "class_metrics.csv")
    df.to_csv(csv_path, index=False)