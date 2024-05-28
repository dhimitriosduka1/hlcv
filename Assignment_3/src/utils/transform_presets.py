import torchvision.transforms as transforms


presets = dict(
    CIFAR10=dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        ),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        )
    ),
    #### TODO ####
    # Define different presets here and try them by specifying their name in the config file
    # Note that you usually need the augmentation **only** for training time!
    # E.g. CIFAR10_WithFlip=dict()
    ##############




    #  This one is for Question 4.
    CIFAR10_VGG=dict(
        train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        eval=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    ),

)