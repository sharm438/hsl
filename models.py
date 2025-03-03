import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def load_net(net_name, num_inp, num_out, device):
    if net_name=='lenet':
        model = LeNet()
    elif net_name=='cifar_cnn':
        model = CIFAR_CNN(num_classes=num_out)
    elif net_name=='agnews_net':
        model = SmallTransformer(vocab_size=95812)

    model.to(device)
    return model


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4, max_seq_len=207, n_heads=4, ffn_dim=512, n_layers=2, dropout=0.1):
        super(SmallTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True  # Enable batch-first input
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.to(torch.long)
        # Embedding and positional encoding
        positional_encoding = self.positional_encoding[:, :x.size(1), :]
        x = self.embedding(x) + positional_encoding[:, :x.size(1), :]
        # Transformer encoder
        encoded = self.transformer_encoder(x)
        # Use the mean of all token embeddings (similar to global average pooling)
        out = encoded.mean(dim=1)
        # Fully connected layer
        out = self.fc(out)
        return out


class LeNet(nn.Module):
    """Simple LeNet model for MNIST."""
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,padding=2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.max_pool2d(x,2)
        x=F.relu(self.conv2(x))
        x=F.max_pool2d(x,2)
        x=x.view(-1,16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

class CIFAR_CNN(nn.Module):
    """Basic CNN for CIFAR-10."""
    def __init__(self, num_classes=10):
        super(CIFAR_CNN,self).__init__()
        self.conv1=nn.Conv2d(3,32,3,padding=1)
        self.conv2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(64*16*16,256)
        self.fc2=nn.Linear(256,num_classes)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(-1,64*16*16)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
