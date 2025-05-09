export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATASET_NAME="Ryan-sjtu/celebahq-caption"

accelerate launch --mixed_precision="bf16" pipelines/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=8 \
  --num_train_epochs=10 --checkpointing_steps=5000 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="test_lora-model" \
  --validation_prompt="sexy woman with big jugs" --report_to="wandb" 
  