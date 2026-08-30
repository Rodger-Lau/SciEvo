import torch.nn as nn
from src.base.model import BaseModel

class HA(BaseModel):
    def __init__(self, **args):
        super(HA, self).__init__(**args)   
        self.fake = nn.Linear(1, 1)

    
    def forward(self, input, label=None):  # (b, t, n, f)
        # Calculate the mean of the last timesteps
        mean_last = input.mean(dim=1, keepdim=True)  # Keeping the dimensions for broadcasting
        # Expand this mean to predict the next 'self.horizon' timesteps
        x = mean_last.expand(-1, self.horizon, -1, -1)
        return x