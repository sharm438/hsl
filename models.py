import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import torchvision.models as models

def load_net(net_name, num_inp, num_out, device):
    if net_name=='lenet':
        model = LeNet()
    elif net_name == 'femnist_cnn':
        model = FEMNIST_CNN(num_out) 
    elif net_name=='cifar_cnn':
        #model = CIFAR_CNN(num_classes=num_out)
        model = ResNet20(num_classes=num_out)
    elif net_name=='agnews_net':
        model = SmallTransformer(vocab_size=95812)
        #model = TextRNN(vocab_size=95812, embed_dim=128, hidden_dim=256, num_classes=num_out)
        #model = AGNewsNet(vocab_size=95812, num_classes=num_out)
    elif net_name == 'resnet18':
        model = models.resnet18()
        # Finetune Final few layers to adjust for tiny imagenet input
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 200)
        model = model.to(device)
    elif net_name == 'resnet50':
        model = models.resnet50(num_classes=100)
    model.to(device)
    return model

class FEMNIST_CNN(nn.Module):
    """
    CNN matching the Leaf TensorFlow model:
    - Conv1: 32 filters, 5x5, padding=2, ReLU
    - Pool1: 2x2 max pool
    - Conv2: 64 filters, 5x5, padding=2, ReLU
    - Pool2: 2x2 max pool
    - FC1: 2048 units, ReLU
    - Output: num_classes units (logits)
    """
    def __init__(self, num_classes=62):
        super(FEMNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Input x: [batch, 1, 28, 28]
        x = F.relu(self.conv1(x))      # [batch, 32, 28, 28]
        x = self.pool(x)               # [batch, 32, 14, 14]
        x = F.relu(self.conv2(x))      # [batch, 64, 14, 14]
        x = self.pool(x)               # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)      # [batch, 64*7*7]
        x = F.relu(self.fc1(x))        # [batch, 2048]
        logits = self.fc2(x)           # [batch, num_classes]
        return logits


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


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = x.to(torch.long)
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        _, (hidden, _) = self.rnn(embedded)  # hidden => [1, batch_size, hidden_dim]
        out = self.fc(hidden.squeeze(0))  # [batch_size, num_classes]
        return out



class AGNewsNet(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_classes=4, padding_idx=0):
        """
        Args:
          vocab_size: the total number of unique token indices (including special tokens).
          embed_dim : dimensionality of the embeddings.
          num_classes: number of output classes (4 for AG_NEWS).
          padding_idx: index to treat as padding (usually 0).
        """
        super(AGNewsNet, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx
        )
        # We'll do a simple average pooling across the sequence dimension,
        # then a fully-connected layer to produce logits.
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Forward pass:
          x shape: [batch_size, seq_len]
        Returns:
          out shape: [batch_size, num_classes]
        """
        x = x.to(torch.long)
        # Embedding layer => [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)

        # Average embedding across the seq_len dimension => [batch_size, embed_dim]
        avg_embed = embedded.mean(dim=1)

        # Classifier => [batch_size, num_classes]
        out = self.fc(avg_embed)
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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A from the paper (average-pool + zero-pad) works too,
            # but 1×1 conv is simpler and widely used today.
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # 3 blocks per layer for depth-20
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc      = nn.Linear(64, num_classes)

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)