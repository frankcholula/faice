from dataset import MNIST
from torch.utils.data import DataLoader

dataset = MNIST()

BATCH_SIZE = 300
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                        persistent_workers=True)  # 数据加载器

for imgs, labels in dataloader:
    print('labels', labels)
    print('labels.shape', labels.shape)
