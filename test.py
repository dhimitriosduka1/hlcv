import timm

base_model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=7)
print(base_model)