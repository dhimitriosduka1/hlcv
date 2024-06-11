import torchvision.transforms as transforms

def get_default_tranforms():
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

def get_geo_transforms():
    return [
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2)
    ]

def get_col_transforms():
    return [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.RandomEqualize(p=0.4),
    ]

presets = dict(
    CIFAR10=dict(
        train=transforms.Compose(get_default_tranforms()),
        eval=transforms.Compose(get_default_tranforms())
    ),
    #  This one is for Question 4.
    CIFAR10_VGG=dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    ),
)

presets["CIFAR10_geo_aug"] = dict(
    train=transforms.Compose(get_geo_transforms() + get_default_tranforms()),
    eval=transforms.Compose(get_default_tranforms())
)

presets["CIFAR10_col_aug"] = dict(
    train=transforms.Compose(get_col_transforms() + get_default_tranforms()),
    eval=transforms.Compose(get_default_tranforms())
)

presets["CIFAR10_geo_col_aug"] = dict(
    train=transforms.Compose(get_geo_transforms() + get_col_transforms() + get_default_tranforms()),
    eval=transforms.Compose(get_default_tranforms())
)