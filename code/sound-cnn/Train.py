import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import pdb

from Dataset import Dataset

class Net(nn.Module):
    def __init__(self):
        """HI
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, (1, 4))
        self.conv2 = nn.Conv2d(3, 15, (1, 4))
        self.conv3 = nn.Conv2d(15, 60, (1, 4))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(211200, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        

    def forward(self, x):
        """
        3 conv blocks -> flatten -> 3 fully connected layers
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 4))
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 4))
        x = F.max_pool2d(F.relu(self.conv3(x)), (1, 4))
        
        # Flatten
        x = x.view(-1, 211200)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    ds = Dataset()
    ds.open()
    sliced = ds.tensor
    y_target = torch.randn(sliced.size(0), 10) # TODO: change to input from SVD
    
    net = Net()
    
    # Define optimizer and loss criterion
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.5)

    # temporary placeholder for 1 batch of data    
    batch_size = 20
    indices = torch.randperm(sliced.size(0)).long().split(batch_size)
    inputs = torch.zeros(batch_size, 1, 320, 768)
    targets = torch.zeros(batch_size, 10)
    
    # Training Here
    for i in range(10):
        print("Epoch {}".format((i+1)))
        for ind in tqdm(indices):
            x = Variable(inputs.copy_(sliced.index(ind)))
            y = Variable(targets.copy_(y_target.index(ind)))
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        print("Loss for epoch {}: {}".format(i+1,loss.data[0]))
    print(net.conv1.weight)

    #Do Testing
        
    
