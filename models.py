import torch
import torch.nn as nn
import torch.nn.functional as F

def load_net(net_name, num_inp, num_out, device):
    if net_name=='lenet':
        model = LeNet()
    elif net_name=='cifar_cnn':
        model = CIFAR_CNN(num_classes=num_out)
    # ---- NEW FOR CITYSCAPES ----
    elif net_name=='cityscapes_seg':
        # A toy segmentation model (extremely simplistic). 
        model = SimpleSegModel(num_classes=num_out)
    else:
        raise ValueError(f"Unknown net_name={net_name}")

    model.to(device)
    return model

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

# ---- NEW FOR CITYSCAPES ----
class SimpleSegModel(nn.Module):
    """
    A minimal segmentation model for demonstration:
      - 2 conv layers
      - directly outputs [N, num_classes, H, W]
    Obviously insufficient for real Cityscapes performance,
    but demonstrates how to integrate a segmentation net.
    """
    def __init__(self, num_classes=20):
        super(SimpleSegModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # [N, 16, H, W]
        x = self.conv2(x)          # [N, num_classes, H, W]
        return x
