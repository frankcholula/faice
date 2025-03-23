# fAIce
EEEM068 Spring 2025 Applied Machine Learning Project: Human Faces Generation with Diffusion Models

## Code Layout
```bash
faice/code
├── Makefile
├── Training_Diffusion_Models.ipynb
├── datasets
│   └── celeba_hq_split
│       ├── test
│       └── train
├── ddpm-butterflies-128
│   ├── logs
│   ├── model_index.json
│   ├── samples
│   ├── scheduler
│   └── unet
├── ddpm-celebahq-256
│   ├── logs
│   ├── model_index.json
│   ├── samples
│   ├── scheduler
│   └── unet
├── diffusion_pipeline.py
├── poetry.lock
├── pyproject.toml
├── train_butterfly.py
├── train_face.py
├── wandb
└── .env
```

## Dataset
As seen in the code layout above, please download the attached dataset `celeba_hq_split.zip` from the email and extract it into the `datasets` folder in order to run the code.

## Credentials
Please use the provided API key and entity in the `.env` file in order to store the runs.
**DO NOT COMMIT THE CREDENTIALS.**