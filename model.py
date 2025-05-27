import torch.nn as nn
import torch
import torchvision.transforms.functional as TF

class ConvPiece(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvPiece, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class DrowsinessModel(nn.Module):

    def __init__(self, in_channel, out_channel, features=[16, 32, 64, 128], input_size=(145, 145)):
        super(DrowsinessModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.3)

        for feature in features:
            self.conv_layers.append(ConvPiece(in_channels=in_channel, out_channels=feature))
            in_channel=feature
        
        with torch.no_grad():
            dummy = torch.zeros(1, 3, *input_size)  
            for conv in self.conv_layers:
                dummy = self.pool(conv(dummy))
            self.flattened_size = dummy.view(1, -1).shape[1]
    
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flattened_size, features[-1])
        self.fc2 = nn.Linear(features[-1], out_channel)
    
    def forward(self, x):

        for conv in self.conv_layers:
            x = self.pool(conv(x))
        
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        return self.fc2(x)


