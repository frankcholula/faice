# Standard library imports
import os
import wandb
from dotenv import load_dotenv

load_dotenv()


class WandBLogger:
    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.is_initialized = False

    @staticmethod
    def login():
        if "WANDB_API_KEY" in os.environ:
            try:
                wandb.login(key=os.environ.get("WANDB_API_KEY"))
                print("Successfully logged into W&B.")
            except Exception as e:
                print(f"Error logging into W&B: {e}")
        else:
            print(
                "WANDB_API_KEY not found in environment variables. Skipping W&B login."
            )

    def setup(self, model=None):
        if not self.config.use_wandb or not self.accelerator.is_main_process:
            return

        wandb.init(
            entity=os.environ.get("WANDB_ENTITY"),
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.num_epochs,
                "train_batch_size": self.config.train_batch_size,
                "image_size": self.config.image_size,
                "seed": self.config.seed,
                "dataset": self.config.dataset_name,
                "model_architecture": self.config.model,
                "scheduler": self.config.scheduler,
            },
        )
        if model is not None and self.config.wandb_watch_model:
            wandb.watch(model, log="all", log_freq=10)
        self.is_initialized = True

    def log_step(self, logs, global_step):
        if not self.config.use_wandb or not self.accelerator.is_main_process:
            return

        wandb.log(logs, step=global_step)

    def log_fid_score(self, fid_score):
        if (
            not self.config.use_wandb
            or not self.accelerator.is_main_process
            or fid_score is None
        ):
            return

        wandb.run.summary["fid_score"] = fid_score

    def finish(self):
        """Clean up and finish the wandb run."""
        if (
            self.is_initialized
            and self.config.use_wandb
            and self.accelerator.is_main_process
        ):
            wandb.finish()
