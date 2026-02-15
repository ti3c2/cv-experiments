import typing as tp
import logging
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    path_root: Path = Path(__file__).parents[1]
    path_data: Path = path_root / "data"
    path_logs: Path = path_root / "logs"

    cuda_visible_devices: str = "0"
    tensorboard_logger_name: str = "lightning_logs"
    checkpoint_filename: str = "epoch{epoch:02d}-val_loss{val_loss:.4g}"
    checkpoint_monitor: str = "loss/val"
    checkpoint_mode: str = "min"
    checkpoint_save_top_k: int = 3
    checkpoint_every_n_epochs: int = 1
    checkpoint_save_last: bool = True
    trainer_max_epochs: int = 100
    trainer_precision: str = "16-mixed"
    trainer_accelerator: str = "gpu"
    trainer_devices: int = 1
    trainer_log_every_n_steps: int = 10

    log_level: int = logging.INFO

    model_config = SettingsConfigDict(
        env_file=path_root / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

    def get_trainer_config(self) -> dict[str, tp.Any]:
        return {
            "max_epochs": self.trainer_max_epochs,
            "precision": self.trainer_precision,
            "accelerator": self.trainer_accelerator,
            "devices": self.trainer_devices,
            "log_every_n_steps": self.trainer_log_every_n_steps,
        }


settings = Settings()