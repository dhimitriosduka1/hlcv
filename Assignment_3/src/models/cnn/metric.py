import torch

class TopKAccuracy:
    def __init__(self, k=1):
        self.k = k

    def compute(self, output, target):
        pred = torch.topk(output, self.k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(self.k):
            correct += torch.sum(pred[:, i] == target).item()
        
        return correct / len(target)
    
    def __str__(self):
        return f"top{self.k}"