import torchvision.transforms as T
def build_transforms(config):

 # build train transformations
    transform_train = [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
    ]

    if RHFlipg:
        transform_train += [
            T.RandomHorizontalFlip()
        ]
    transform_train += [T.ToTensor()]
    if blur:
        transform_train += [T.GaussianBlur()]

    return transform_train