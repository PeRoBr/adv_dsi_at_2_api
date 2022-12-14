import torch
import torch.nn as nn
import torch.nn.functional as F

class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 10)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        return self.layer_out(x)

# model = PytorchMultiClass(14)
# model.load_state_dict(torch.load("../models/pytorch_beer_classifier.pt"))