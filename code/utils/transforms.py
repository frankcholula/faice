import torchvision.transforms as T
def build_transforms(config):

 # build train transformations
    transform_train = [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
    ]

    if RHFlip:
        transform_train += [
            T.RandomHorizontalFlip()
        ]
    transform_train += [T.ToTensor()]
    if blur:
        transform_train += [T.GaussianBlur()]


    return transform_train