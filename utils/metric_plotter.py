"""Save and refresh one training metrics figure during training."""

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_top1",
    "val_top1",
    "train_top5",
    "val_top5",
    "lr",
    "train_fps",
    "val_fps",
)


class MetricPlotter:
    def __init__(self, output_dir: str, filename: str = "training_metrics.png", resume: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.image_path = self.output_dir / filename
        self.csv_path = self.output_dir / "training_metrics.csv"
        self.best_metrics_path = self.output_dir / "best_metrics.json"
        self.history = []
        self.best_metrics = None
        if resume:
            self._load_csv()
            self._load_best_metrics()

    def update(self, metrics: dict, best_metrics: dict | None = None):
        row = {column: metrics.get(column) for column in METRIC_COLUMNS}
        self.history.append(row)
        if best_metrics is not None:
            self.best_metrics = best_metrics
            self._write_best_metrics()
        self._write_csv()
        self._write_plot()
        print(f"### Metrics Plot Updated: {self.image_path} ###")

    def _write_csv(self):
        tmp_path = self.csv_path.with_suffix(".csv.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerows(self.history)
        os.replace(tmp_path, self.csv_path)

    def _write_best_metrics(self):
        if self.best_metrics is None:
            return
        tmp_path = self.best_metrics_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(self.best_metrics, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(tmp_path, self.best_metrics_path)

    def _load_best_metrics(self):
        if not self.best_metrics_path.exists():
            return
        with open(self.best_metrics_path, "r", encoding="utf-8") as file:
            self.best_metrics = json.load(file)

    def _load_csv(self):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                clean_row = {}
                for column in METRIC_COLUMNS:
                    value = row.get(column)
                    if value in (None, ""):
                        clean_row[column] = None
                    elif column == "epoch":
                        clean_row[column] = int(float(value))
                    else:
                        clean_row[column] = float(value)
                self.history.append(clean_row)

    def _series(self, key):
        return [row.get(key) for row in self.history]

    def _best_epoch(self):
        if not self.best_metrics:
            return None
        return self.best_metrics.get("best_epoch")

    def _draw_best_epoch(self, axis):
        best_epoch = self._best_epoch()
        if best_epoch is None:
            return
        axis.axvline(best_epoch, color="#D32F2F", linestyle="--", linewidth=1.2, alpha=0.8)

    def _plot_pair(self, axis, epochs, train_key, val_key, title, ylabel):
        train_values = self._series(train_key)
        val_values = self._series(val_key)
        axis.plot(epochs, train_values, marker="o", linewidth=1.8, label="train")
        if any(value is not None for value in val_values):
            axis.plot(epochs, val_values, marker="o", linewidth=1.8, label="val")
        self._draw_best_epoch(axis)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()

    def _write_plot(self):
        epochs = self._series("epoch")
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle("Training Metrics", fontsize=16)

        self._plot_pair(axes[0, 0], epochs, "train_loss", "val_loss", "Loss", "loss")
        self._plot_pair(axes[0, 1], epochs, "train_top1", "val_top1", "Top1", "accuracy (%)")
        self._plot_pair(axes[1, 0], epochs, "train_top5", "val_top5", "Top5", "accuracy (%)")

        axes[1, 1].plot(epochs, self._series("lr"), marker="o", linewidth=1.8, color="#7E57C2")
        axes[1, 1].set_title("Learning Rate")
        axes[1, 1].set_xlabel("epoch")
        axes[1, 1].set_ylabel("lr")
        axes[1, 1].grid(True, alpha=0.25)
        self._draw_best_epoch(axes[1, 1])

        self._plot_pair(axes[2, 0], epochs, "train_fps", "val_fps", "FPS", "images/sec")
        axes[2, 1].axis("off")
        if self.best_metrics:
            lr_steps = self.best_metrics.get("lr_steps", [])
            if isinstance(lr_steps, list):
                lr_steps = ",".join(str(step) for step in lr_steps) or "[]"
            summary = "\n".join(
                [
                    "Best Metrics",
                    f"epoch: {self.best_metrics.get('best_epoch')}",
                    f"val_top1: {self.best_metrics.get('best_top1', 0.0):.4f}",
                    f"val_top5: {self.best_metrics.get('best_top5', 0.0):.4f}",
                    f"model: {self.best_metrics.get('model', '')}",
                    f"loss: {self.best_metrics.get('loss', '')}",
                    f"optimizer: {self.best_metrics.get('optimizer', '')}",
                    f"weight_decay: {self.best_metrics.get('weight_decay', '')}",
                    f"lr: {self.best_metrics.get('lr', '')}",
                    f"lr_policy: {self.best_metrics.get('lr_policy', '')}",
                    f"warmup_length: {self.best_metrics.get('warmup_length', '')}",
                    f"lowest_lr: {self.best_metrics.get('lowest_lr', '')}",
                    f"lr_steps: {lr_steps}",
                    f"lr_gamma: {self.best_metrics.get('lr_gamma', '')}",
                ]
            )
            axes[2, 1].text(0.02, 0.95, summary, va="top", ha="left", fontsize=10, family="monospace")

        fig.tight_layout(rect=(0, 0, 1, 0.97))
        tmp_path = self.image_path.with_suffix(".png.tmp")
        fig.savefig(tmp_path, dpi=150, format="png")
        plt.close(fig)
        os.replace(tmp_path, self.image_path)
