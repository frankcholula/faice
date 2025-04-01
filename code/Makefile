.PHONY: trainb
trainb:
	python main.py --dataset butterfly

.PHONY: trainf
trainf:
	python main.py --dataset face

.PHONY: tensorboardb
tensorboardb:
	tensorboard --logdir=ddpm-butterflies-128/logs

.PHONY: clean
clean:
	rm -rf runs/ddpm-celebhq-256/logs/train_example/