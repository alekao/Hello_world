import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision

# Data preprocessing -- convert data to tensor then normalizing them
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# --Download MNIST dataset and apply data preprocessing--#
trainset = datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)

# --load data and split them into batches--#
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

# --Build a neural network--#
class Net(nn.Module):
    def __init__(self):
        # --Inherit the functions and variables
        # --from the PyTorch nerual network module
        # --and initialize them for this network
        super(Net, self).__init__()

        # --Add layers to the network!
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2)
        self.linear = nn.Linear(32*5*5, 10)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.output = nn.Softmax()

    # --define the forward function
    # --so that simply doing Net(input)
    # --would return the output of the NN
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pooling(x)
        x = x.view(-1, 32*5*5)
        x = self.linear(x)
        return self.output(x)

#--create neural network
net = Net()

#--define loss and optimizer
loss_fn = nn.CrossEntropyLoss()
opti = torch.optim.Adam(net.parameters(),lr=2)

#--Train
for epoch in range(10):
    running_loss = 0.0
    running_accuracy = 0
    for image, label in trainloader:
        net.zero_grad()

        #--forward pass
        output = net(image)

        #--compute the loss
        loss = loss_fn(output,label)

        #--compute the gradient for each parameter
        loss.backward()

        #--update the parameters with
        #--the given learning rate
        opti.step()

        #--Add the losses and accuracy
        running_loss+=loss.item()
        _, pred = torch.max(output, 1)
        running_accuracy+=(pred==label).sum().item()
    print('Epoch {} -- Loss: {:.4f}, accuracy: {:.4f}'.format(epoch, running_loss/len(trainloader),
                                                      float(running_accuracy)/len(trainloader)))

torch.save(net, 'MNIST_network.pth')
      
        
