import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_layers=[512, 256], use_dropout=False, use_batch_norm=False):
        super(MLP, self).__init__()
        
        layers = []
        in_features = input_size
        
        for h in hidden_layers:
            layers.append(nn.Linear(in_features, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.2))
            in_features = h
            
        layers.append(nn.Linear(in_features, 10)) # 10 classes for MNIST
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input image
        x = x.view(x.size(0), -1)
        return self.model(x)
