# fAIce 🫥
EEEM068 Spring 2025 Applied Machine Learning Project: Human Faces Generation with Diffusion Models

## Code Layout
```bash
faice/code
├── Makefile
├── Training_Diffusion_Models.ipynb
├── args.py
├── conf
│   └── training_config.py
├── datasets
│   └── celeba_hq_split
├── main.py
├── models
│   └── unet.py
├── pipelines
│   └── ddpm.py
├── runs
│   ├── ddpm-face-10
│   └── ddpm-butterfly-10
├── tests
├── utils
│   ├── loggers.py
│   ├── metrics.py
│   └── training.py
└── wandb
```
## Running the Experiments
Pleas see the `experiments` folder for running the experments.
You have the option to use the `Makefile` as well. 
```bash
python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name baseline \
    --calculate_fid \
    --calculate_is \
    --verbose
```
Please use the `--verbose` flag to check your parameters before running the experiments.

⚠️ **Unless you're running hyperparameter tuning, please make sure yoru experiement batch size is consistent for the ablation study**
1. If you're running the experiments on `Otter`, please lock the batch size to `24` for memory reasons. 
2. If you're running the experiments on `Eureka2`, please set the batch size to `64` for faster training.

## Dataset
As seen in the code layout above, please download the attached dataset `celeba_hq_split.zip` from the email and extract it into the `datasets` folder in order to run the code.

## Credentials
Please use the provided API key and entity in the `.env` file in order to store the runs on Weights & Biases.
```bash
# sample .env file WANDB_ENTITY=<your_wandb_entity>
WANDB_API_KEY=<your_wandb_api_key>
```
🚨 **DO NOT COMMIT THE CREDENTIALS**
