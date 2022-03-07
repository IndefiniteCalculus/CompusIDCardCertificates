import torch.nn as nn
import torch.nn.functional as F
import torch


class ChineseCharNet(nn.Module):
    def __init__(self,num_class):
        super(ChineseCharNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            # nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(3200,
                      1600),
            # nn.Linear(512, 512),
            nn.Linear(1600, num_class)
        )
        # self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1,self.num_flat_features(x))
        x = self.fc(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__ == "__main__":
    model = ChineseCharNet(3981)
    model.forward(torch.ones((1,1,40,40)))