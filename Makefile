.PHONY: trainb
trainb:
	python code/train_butterfly.py

.PHONY: trainf
trainf:
	python code/train_face.py

.PHONY: tensorboardb
tensorboardb:
	tensorboard --logdir=code/ddpm-butterflies-128/logs

.PHONY: clean
clean:
	rm -rf code/ddpm-celebhq-256/logs/train_example/