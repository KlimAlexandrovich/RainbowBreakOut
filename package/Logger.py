import os
import torch
import time
import csv
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from IPython.display import clear_output
from utils import get_last_update


class SmartLogger:
    """ A utility class for logging model weights and scalar metrics during training. """

    def __init__(self, *model_names: str, log_dir: str, exp_name: str):
        """ Initialize the Logger with a directory path and experiment name. """
        self.path: str = os.path.join(log_dir, exp_name)
        self.checkpoint_path: str = os.path.join(self.path, "checkpoint")
        self.model_paths: dict[str, str] = {name: os.path.join(self.checkpoint_path, name) for name in model_names}
        self.scalars_path: str = os.path.join(self.path, "scalars")
        self._weights_ext: str = "pt"
        self.init()

    def init(self) -> None:
        """ Create the necessary directories for checkpoints and scalars. """
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.scalars_path, exist_ok=True)
        for model in self.model_paths.keys():
            os.makedirs(os.path.join(self.checkpoint_path, model), exist_ok=True)

    def checkpoint(self, weights: dict, model: str) -> None:
        """ Save model weights as a checkpoint file. """
        assert model in self.model_paths.keys(), f"Logger does not know model: {model}."
        torch.save(weights, os.path.join(self.model_paths[model], f"weights_{time.time()}.{self._weights_ext}"))

    def set_scalars(self, **scalars: int | float) -> None:
        """ Log multiple scalar values with the current timestamp. """
        for name, value in scalars.items():
            self._set_scalar(time.time(), name, value)

    def _set_scalar(self, dtime: float | int, name: str, value: int | float | str) -> None:
        """ Log a single scalar value to a CSV file. """
        file_path: str = os.path.join(self.scalars_path, f"{name}.csv")
        file_exists: bool = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(["dtime", "value"])
            writer.writerow([dtime, value])

    def get_last_update(self, model: str) -> Optional[str]:
        """ Retrieve the path to the most recent checkpoint file in the directory. """
        assert model in self.model_paths.keys(), f"Logger does not know model: {model}."
        return get_last_update(self.model_paths[model])

    def draw_scalars(self, exclude: Optional[list[str]] = None) -> None:
        """ Plots logged metrics from CSV files.
            Automatically creates subplots for each metric found in the scalar's directory. """
        assert os.path.exists(self.scalars_path), f"Directory {self.scalars_path} does not exist."
        csv_files: list[str] = [file for file in os.listdir(self.scalars_path) if file.endswith(".csv")]
        if exclude is not None:
            # TODO: Implement exclude argument.
            raise NotImplementedError("Not implemented exclude argument yet.")
        if len(csv_files) == 0:
            print("No CSV files found.")
        else:
            clear_output(wait=True)
            n_files: int = len(csv_files)
            fig, axes = plt.subplots(n_files, 1, figsize=(5, 4 * n_files), sharex=False)
            axes = [axes] if (n_files == 1) else axes
            for idx, file in enumerate(csv_files):
                file_path: str = os.path.join(self.scalars_path, file)
                df: pd.DataFrame = pd.read_csv(file_path)
                metric_name: str = file.replace(".csv", "")
                # Drawing.
                axes[idx].plot(df["dtime"].index, df["value"], label=metric_name, color="dodgerblue")
                axes[idx].set_title(f"Metric: {metric_name}")
                axes[idx].set_xlabel("Epoch")
                axes[idx].set_ylabel("Value")
                axes[idx].legend()
                axes[idx].grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
