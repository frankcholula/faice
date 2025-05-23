# Standard library imports
import os
import time
from contextlib import contextmanager

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
            entity=self.config.wandb_entity,
            project=self.config.wandb_project,
            name=self.config.wandb_run_name,
            config={
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "train_batch_size": self.config.train_batch_size,
                "image_size": self.config.image_size,
                "seed": self.config.seed,
                "dataset": self.config.dataset_name,
                "model": self.config.model,
                "unet_variant": self.config.unet_variant,
                "layers_per_block": self.config.layers_per_block,
                "base_channels": self.config.base_channels,
                "multi_res": self.config.multi_res,
                "attention_head_dim": self.config.attention_head_dim,
                "upsample_type": self.config.upsample_type,
                "downsample_type": self.config.downsample_type,
                "scheduler": self.config.scheduler,
                "eta": self.config.eta,
                "pipeline": self.config.pipeline,
                "prediction_type": self.config.prediction_type,
                "rescale_betas_zero_snr": self.config.rescale_betas_zero_snr,
                "loss_type": self.config.loss_type,
                "use_lpips": self.config.use_lpips,
                "condition_on": self.config.condition_on,
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

    def log_fid_score_rec(self, fid_score):
        if (
            not self.config.use_wandb
            or not self.accelerator.is_main_process
            or fid_score is None
        ):
            return

        wandb.run.summary["fid_score_reconstruction"] = fid_score

    def log_inception_score(self, inception_score):
        if (
            not self.config.use_wandb
            or not self.accelerator.is_main_process
            or inception_score is None
        ):
            return

        wandb.run.summary["inception_score"] = inception_score

    def save_model(self):
        if not self.config.use_wandb or not self.accelerator.is_main_process:
            return
        artifact = wandb.Artifact(
            name=f"{self.config.wandb_run_name}",
            type="model",
            metadata={
                "model": self.config.model,
                "num_epochs": self.config.num_epochs,
                "scheduler": self.config.scheduler,
                "dataset": self.config.dataset_name,
                "image_size": self.config.image_size,
                "beta_schedule": self.config.beta_schedule,
            },
        )
        try:
            artifact.add_file(
                os.path.join(
                    self.config.output_dir,
                    self.config.model,
                    "diffusion_pytorch_model.safetensors",
                )
            )
            artifact.add_file(
                os.path.join(self.config.output_dir, self.config.model, "config.json")
            )
            artifact.add_file(os.path.join(self.config.output_dir, "model_index.json"))
            artifact.add_file(
                os.path.join(self.config.output_dir, "scheduler/scheduler_config.json")
            )

            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"Error saving model artifact to W&B: {e}")

    def finish(self):
        """Clean up and finish the wandb run."""
        if (
            self.is_initialized
            and self.config.use_wandb
            and self.accelerator.is_main_process
        ):
            wandb.finish()


@contextmanager
def timer(msg="all tasks"):
    """
    Calculate the time of running
    @return:
    """
    startTime = time.time()
    yield
    endTime = time.time()
    # print(f'The time cost for {msg}：{round(1000.0 * (endTime - startTime), 2)}, ms')
    print(f"The time cost for {msg}：", round((endTime - startTime) / 60, 2), "minutes")
