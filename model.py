import torch.nn as nn
import torchvision


# model = Net()
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.in_features = self.model.fc.in_features
        self.layer = nn.Linear(self.in_features, 43)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.in_features)
        x = self.layer(x)
        return x
