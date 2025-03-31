import wandb

class WandBLogger:
    def __init__(self, config, accelerator): 
        self.config = config
        self.accelerator = accelerator
        self.is_initialized = False


    def setup(self, model=None):
        if not self.config.use_wandb or not self.accelerator.is_main_process:
            return
        
        wandb.init()