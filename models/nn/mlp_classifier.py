from torch.nn import Linear, Module
from torch.nn.functional import dropout

from torch import arange, Tensor
from torch.nn import Linear, Sequential
from torch.nn.functional import dropout

class MLPClassifier(Module):
    def __init__(self, hidden_channel_dimensions:list,  num_classes:int):
        super(MLPClassifier, self).__init__()
        self.layers = Sequential()
        for i in range(1,len(hidden_channel_dimensions)):
            self.layers.append(
                Linear(in_features=hidden_channel_dimensions[i-1], out_features=hidden_channel_dimensions[i])
            )
        self.linear = Linear(hidden_channel_dimensions[-1], num_classes)

    def forward(self, features:Tensor, dropout_percentage:float=0.5):
        x = features
        for layer in self.layers:
            x = layer(x).relu()
            
        x = dropout(x, p=dropout_percentage, training=self.training)
        x = self.linear(x)
        
        return x
