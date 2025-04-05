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
## Running the experiments
Sample command posted below. Consult the `Makefile` for other available commands.
```bash
python main.py --dataset face --num_epochs 1 --calculate_fid --calculate_is --verbose
```

## Dataset
As seen in the code layout above, please download the attached dataset `celeba_hq_split.zip` from the email and extract it into the `datasets` folder in order to run the code.

## Credentials
Please use the provided API key and entity in the `.env` file in order to store the runs on Weights & Biases.
```bash
# sample .env file
WANDB_ENTITY=<your_wandb_entity>
WANDB_API_KEY=<your_wandb_api_key>
```
**DO NOT COMMIT THE CREDENTIALS.**
