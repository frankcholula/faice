.PHONY: trainb
trainb:
	python train_butterfly.py

.PHONY: trainf
trainf:
	python train_face.py

.PHONY: tensorboardb
tensorboardb:
	tensorboard --logdir=ddpm-butterflies-128/logs

.PHONY: clean
clean:
	rm -rf ddpm-celebhq-256/logs/train_example/