import torchvision.transforms as T
def build_transforms(config):

 # build train transformations
    transform_train = [
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
    ]

    if config.RHFlip:
        transform_train += [
            T.RandomHorizontalFlip()
        ]
    transform_train += [T.ToTensor()]
    if config.gblur:
        transform_train += [T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]


    return transform_train