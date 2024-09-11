import torch
from model.framework.base import Net

class CSIID(Net):
    def __init__(self, model_cfg):
        super().__init__(model_cfg);

        # First convolutional layer
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 30, kernel_size=(100, 3), stride=(1, 3)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # Second convolutional layer
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, kernel_size=(5, 3), stride=(2, 1), padding=(2, 1)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # Third convolutional layer
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(30, 30, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0)),
            torch.nn.BatchNorm2d(30),
            torch.nn.ReLU()
        )

        # LSTM layer
        self.lstm = torch.nn.LSTM(input_size=480, hidden_size=128, num_layers=1, batch_first=True)

        # Fully connected layer
        self.fc = torch.nn.Linear(128, self.known_p_num)

        # Softmax
        self.softmax = torch.nn.Softmax(dim=1)

    def p_classify(self, x):

        x = x.permute(0, 2, 1, 3)

        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Crop to 203*22*30


        # Reshape for LSTM input (batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)

        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output from the last time step

        # Fully connected layer
        x = self.fc(x)

        # Softmax
        x = self.softmax(x)

        return x
    
    def cal_loss(self, amps, ids, envs, is_target_data = False):
        id_probs = self.p_classify(amps);
        return torch.nn.CrossEntropyLoss()(id_probs, ids);