import torchvision.transforms as transforms

def get_default_tranforms():
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

def get_default_tranforms_vgg():
    return [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]

def get_geo_transforms():
    return [
        transforms.RandomHorizontalFlip(p=0.6),
        transforms.RandomRotation(15),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2)
    ]

def get_col_transforms():
    return [
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        ], p=0.5),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomEqualize(p=0.5)
    ]

presets = dict(
    CIFAR10=dict(
        train=transforms.Compose(get_default_tranforms()),
        eval=transforms.Compose(get_default_tranforms())
    ),
    #  This one is for Question 4.
    CIFAR10_VGG=dict(
        train=transforms.Compose(get_default_tranforms_vgg()),
        eval=transforms.Compose(get_default_tranforms_vgg())
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

presets["CIFAR10_geo_col_aug_aa_policy"] = dict(
    train=transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    ]),
    eval=transforms.Compose([])
)

presets["CIFAR10_VGG_HF"] = dict(
    train=transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.6),
    ] + get_default_tranforms_vgg()),
    eval=transforms.Compose(get_default_tranforms_vgg())
)

presets["CIFAR10_VGG_ROT"] = dict(
    train=transforms.Compose([
        transforms.RandomRotation(15),
    ] + get_default_tranforms_vgg()),
    eval=transforms.Compose(get_default_tranforms_vgg())
)

presets["CIFAR10_VGG_PERSP"] = dict(
    train=transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.2)
    ] + get_default_tranforms_vgg()),
    eval=transforms.Compose(get_default_tranforms_vgg())
)

presets["CIFAR10_VGG_GEO_COMBINED"] = dict(
    train=transforms.Compose(get_geo_transforms() + get_default_tranforms_vgg()),
    eval=transforms.Compose(get_default_tranforms_vgg())
)