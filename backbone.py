from torch import nn, Tensor
import torch

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, gru_dropout=0.0, fc_dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers =num_layers

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(p=fc_dropout)

        self.fc_out = nn.Linear(
            in_features=hidden_size,
            out_features=output_size
        )

        self.init_params()

    def init_params(self):
        '''
        权重正交初始化，防止梯度消失/爆炸
        '''
        for name, param in self.rnn.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
    

    def forward(self, x: Tensor, h_0=None):
        device = x.device
        batch_size = x.size(0)  # 必须要 batch_first

        if h_0 is None:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            # 隐藏状态的大小是pytorch预先规定的

        out, _ = self.rnn(x, h_0)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc_out(out)
        return out

