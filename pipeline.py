import os

from cfg import CFG
from metrics import plot_metrics
from model_eval import eval_model, plot_loss_curves
from plot_images import plot_images_from_dataloader, plot_images_from_folder
from training import train_model
from wrong_predictions import get_evaluation_dataframes


def start_pipeline():
    """
    Starts the training and evaluation pipeline for the model.

    The pipeline includes:
    1. Creating necessary directories.
    2. Plotting images from the training and test datasets if specified.
    3. Training the model with early stopping and learning rate scheduling.
    4. Plotting training and validation loss and accuracy curves.
    5. Evaluating the model on the test dataset.
    6. Plotting the confusion matrix and generating evaluation dataframes.

    Example usage:
        start_pipeline()
    """
    if not os.path.exists(CFG.model_folder):
        os.makedirs(CFG.model_folder)

    if CFG.plot_image_at_begin:
        plot_images_from_folder(folder=CFG.train_folder, n_images=4, images_per_row=4)
        plot_images_from_folder(folder=CFG.test_folder, n_images=4, images_per_row=4)

        plot_images_from_dataloader(
            dataloader=CFG.train_dataloader,
            class_names=CFG.class_names,
            n_images=4,
            images_per_row=4,
        )

    results, df = train_model(
        model=CFG.model,
        train_dataloader=CFG.train_dataloader,
        val_dataloader=CFG.val_dataloader,
        optimizer=CFG.optimizer,
        loss_fn=CFG.loss_fn,
        early_stopper=CFG.early_stopper,
        lr_scheduler=CFG.lr_scheduler,
        epochs=CFG.epochs,
        device=CFG.device,
    )

    plot_loss_curves(results=results, model_folder=CFG.model_folder)

    eval_model(
        model=CFG.model,
        dataloader=CFG.test_dataloader,
        loss_fn=CFG.loss_fn,
        df=df,
        device=CFG.device,
        model_folder=CFG.model_folder,
    )

    y_pred, y_true = plot_metrics(
        model=CFG.model,
        dataloader=CFG.test_dataloader,
        dataset=CFG.test_dataset,
        class_names=CFG.class_names,
        device=CFG.device,
        model_folder=CFG.model_folder,
    )

    get_evaluation_dataframes(
        y_pred=y_pred,
        y_true=y_true,
        dataset=CFG.test_dataset,
        class_names=CFG.class_names,
        model_folder=CFG.model_folder,
    )
