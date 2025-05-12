# fAIce 🫥
EEEM068 Spring 2025 Applied Machine Learning Project: Human Faces Generation with Diffusion Models. All model runs can be found at the Weights & Biases link [here](https://wandb.ai/frankcholula/faice?nw=nwusertsufanglus). Planning documentation can be found [here](https://frankcholula.notion.site/faice?pvs=4) along with a web version of the final paper [here](https://frankcholula.notion.site/diffusion-paper?pvs=4).

![Will Smith](code/assets/will_smith.png)

## Code Layout
```bash
faice/code
├── code
│   ├── args.py
│   ├── conf
│   ├── datasets
│   ├── eda
│   ├── experiments
│   ├── main.py
│   ├── Makefile
│   ├── models
│   ├── pipelines
│   ├── runs
│   └── utils
├── docs
├── environment.yml
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
└── Surrey_CSEE_Thesis_Template
```

## Running the Experiments
⚠️ **Please first request for cluster access to `Eureka2` and `Otter` from the CSEE department. Otherwise, there's an `Otter Setup` documentation [here](https://frankcholula.notion.site/otter-setup?pvs=4).**

Please design the experiments according to your tasks and put them in the  `experiments` folder accordingly. You have the option to use the `Makefile` as well. 
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
Given that runs are computationally expensive, I recommend using the `--verbose` flag to check your parameters before running the experiments.

⚠️ **Unless you're running hyperparameter tuning, please make sure yoru experiement batch size is consistent for the ablation study**
1. If you're running the experiments on `Otter`, please lock the batch size to `24` for memory reasons. 
2. If you're running the experiments on `Eureka2`, please set the batch size to `64` for faster training.

After you've run the experiments, please documen the results in the Notion [page](https://frankcholula.notion.site/faice?pvs=4).

## Dataset Preparation
For **unconditional generation**, please download the attached dataset `celeba_hq_split.zip` from the email and extract it into the `datasets` folder in order to run the code.

For **conditional generation**, please download the dataset sent in the WhatsApp group. The layout of your dataset should be as follows:
```bash



## Credentials
Please use the provided API key and entity in the `.env` file in order to store the runs on Weights & Biases.
```bash
# sample .env file
WANDB_ENTITY=<your_wandb_entity>
WANDB_API_KEY=<your_wandb_api_key>
```
🚨 **DO NOT COMMIT THE CREDENTIALS**
