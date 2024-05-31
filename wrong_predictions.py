import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


def plot_wrong_predictions(
    wrong_predictions: pd.DataFrame,
    n_images: int = 20,
    images_per_row: int = 4,
    figsize_width: int = 20,
    fontsize: int = 12,
):
    """
    Plots a specified number of wrong predictions from a DataFrame, displaying each image with its true and predicted class names, and image path below.

    Args:
        wrong_predictions (pd.DataFrame): A DataFrame containing the wrong predictions. It must include columns 'img_path', 'y_true_classname', and 'y_pred_classname'.
        n_images (int, optional): The total number of wrong prediction images to display. Defaults to 20.
        images_per_row (int, optional): The number of images to display per row in the grid layout. Defaults to 4.
        figsize_width (int, optional): The width of the figure to display the images. The height is automatically calculated based on the number of rows. Defaults to 20.
        fontsize (int, optional): The font size of the title for each subplot, which includes the true and predicted class names, and the image path. Defaults to 12.

    Example usage:
        # Assuming `wrong_predictions` is your DataFrame containing the columns 'img_path', 'y_true_classname', and 'y_pred_classname'.
        plot_wrong_predictions(wrong_predictions=wrong_predictions, n_images=20, images_per_row=4, figsize_width=20, fontsize=12)
    """
    n_cols = images_per_row
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize_width, n_rows * 5)
    )
    axes = axes.reshape(-1)

    for i, row in enumerate(wrong_predictions.iloc[:n_images].itertuples()):
        img_path = row.img_path
        true_classname = row.y_true_classname
        pred_classname = row.y_pred_classname

        img = mpimg.imread(img_path)
        axes[i].imshow(img / 255.0)

        axes[i].set_title(
            f"True: {true_classname}, Pred: {pred_classname}\n  img_path: {img_path}",
            fontsize=fontsize,
        )

        axes[i].axis("off")

    for j in range(i + 1, n_rows * n_cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def get_evaluation_dataframes(y_pred, y_true, dataset, class_names, model_folder):
    """
    Generates DataFrames of all predictions and wrong predictions, and saves them as CSV files.

    Args:
        y_pred (torch.Tensor): Predicted labels from the model.
        y_true (torch.Tensor): True labels.
        dataset (Dataset): The dataset containing the file paths.
        class_names (list): List of class names.

    Example usage:
        y_pred = model_predictions
        y_true = true_labels
        dataset = test_dataset
        class_names = ["Class 0", "Class 1"]

        get_evaluation_dataframes(y_pred=y_pred, y_true=y_true, dataset=dataset, class_names=class_names)
    """
    test_filepaths = dataset.get_filepaths()

    df_dict = {
        "img_path": test_filepaths,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_true_classname": [class_names[i] for i in y_true],
        "y_pred_classname": [class_names[i] for i in y_pred],
    }

    pred_df = pd.DataFrame(df_dict)
    pred_df["pred_correct"] = pred_df["y_true"] == pred_df["y_pred"]

    csv_path = os.path.join(model_folder, f"predictions.csv")
    pred_df.to_csv(csv_path, index=False)

    wrong_predictions = pred_df[pred_df["pred_correct"] == False]

    csv_wrong_path = os.path.join(model_folder, "wrong_predictions.csv")
    wrong_predictions.to_csv(csv_wrong_path, index=False)

    plot_wrong_predictions(wrong_predictions)
