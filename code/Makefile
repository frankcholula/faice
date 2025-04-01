.PHONY: trainb
trainb:
	python main.py --dataset butterfly --num_epochs 1

.PHONY: trainf
trainf:
	python main.py --dataset face --num_epochs 1

.PHONY: tensorboardb
tensorboardb:
	tensorboard --logdir=ddpm-butterflies-128/logs

.PHONY: clean
clean:
	rm -rf runs/ddpm-celebhq-256/logs/train_example/