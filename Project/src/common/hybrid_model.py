import torch

class HybridModel(torch.nn.Module):
    def __init__(self, vision_model, num_labels):
        super(HybridModel, self).__init__()
        self.vision_model = vision_model
        self.hand_feature_dim = 21 * 3
        self.fusion_layer = torch.nn.Linear(
            vision_model.config.hidden_size + self.hand_feature_dim, 
            vision_model.config.hidden_size
        )
        self.classifier = torch.nn.Linear(vision_model.config.hidden_size, num_labels)

    def forward(self, pixel_values, hand_features, labels):
        # print(self.vision_model(pixel_values).hidden_states[-1][:, 0, :])
        vision_outputs = self.vision_model(pixel_values).hidden_states[-1][:, 0, :]
        fused_features = self.fusion_layer(
            torch.cat([vision_outputs.to(torch.float64), hand_features.to(torch.float64)], dim=1).to(torch.float64)
        )
        logits = self.classifier(fused_features)
        return logits