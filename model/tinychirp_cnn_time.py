import torch
import torch.nn as nn

DEFAULT_INPUT_SHAPE = (1, 1 ,48000) # (N, C, T), add N=1 for avoid tvm's error on Pool1d
DEFAULT_SLIDING_STEP_SIZE = 16000

class TinyChirpCNNTime(nn.Module):
    def __init__(self, channel1=4, channel2=8):
        super().__init__()

        self.conv1 = nn.Conv1d(1, channel1, kernel_size=3)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.conv2 = nn.Conv1d(channel1, channel2, kernel_size=3)
        self.fc1 = nn.Linear(channel2, 64)
        self.fc2 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.adpool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.adpool(x).squeeze(-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x
    
if __name__ == "__main__":
    x = torch.randn(*DEFAULT_INPUT_SHAPE)
    net = TinyChirpCNNTime().eval()
    y = net(x)
    print(y.shape)