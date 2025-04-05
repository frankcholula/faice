.PHONY: trainb
trainb:
	python main.py --dataset butterfly --num_epochs 10

.PHONY: trainf
trainf:
	python main.py --dataset face --num_epochs 1 --calculate_fid

.PHONY: tensorboardb
tensorboardb:
	tensorboard --logdir=ddpm-butterflies-128/logs

.PHONY: clean
clean:
	rm -rf runs/ddpm-celebhq-256/logs/train_example/

.PHONY: calculate_fid
calculate_fid:
	python -m pytorch_fid runs/ddpm-face-1/fid/real runs/ddpm-face-1/fid/fake --device cuda:0
