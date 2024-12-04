import torch
import torch.nn as nn

class CNN_RNN_Model(nn.Module):
    def __init__(self, num_landmarks=75, hidden_size=128, num_classes=50, dropout_rate=0.5):
        super(CNN_RNN_Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = nn.Linear(num_landmarks // 2 * 128, hidden_size)  # Adjusted for pooling
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, seq_len, num_landmarks, channels = x.size()
        x = x.view(batch_size * seq_len, 1, num_landmarks, channels)

        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        x = x.view(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = x.view(batch_size, seq_len, -1)
        
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]
        
        x = self.fc2(x)
        return x
