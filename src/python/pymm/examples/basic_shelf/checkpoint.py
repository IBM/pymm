#Import Libraries


from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pymm
import numpy as np
import sys

args={}
kwargs={}
args['batch_size']=1000
args['test_batch_size']=1000
args['epochs']=3  #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01 #Learning rate is how fast it will decend. 
args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1 #random seed
args['log_interval']=10
args['cuda']=False




#load the data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)




class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        #Softmax gets probabilities. 
        return F.log_softmax(x, dim=1)




def train(epoch):
    model.train()
    i = 0
    print (type(i))
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        #Variables in Pytorch are differenciable. 
        data, target = Variable(data), Variable(target)
        #This will zero out the gradients for this batch. 
        optimizer.zero_grad()
        output = model(data)
        # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
#        print(optimizer.param_groups[0].keys())
#        for i in range(len(optimizer.param_groups[0]["params"])):
#            print (type((optimizer.param_groups[0]["params"][i])))

##        print ("model")
##        for name, param in model.named_parameters():   
##            print (type(param))
        name = model.named_parameters()
        print(type(model))
        print(issubclass(type(model), nn.Module))
#        print(type(model.named_parameters()) is <class 'generator'>)
        shelf.save({
                    'epoch': np.array([i]),
                    'model' : model,
                    'optimizer' : optimizer,
                    'loss': torch.empty(1) 
        }, header_name = "mnist", is_inplace=False, is_create_empty=False)


#        shelf.dict_load({
#                    'epoch': np.array([i]),
#                    'model' : model,
#                    'loss': np.array([loss.data]) 
#        }, header_name = "mnist", is_torch_save=True, is_create_empty=False)


#        if(i==0):    
#           shelf.torch_save_model(model, "mnist")
#        print (model.conv1.weight)
#        shelf.torch_load_model(model, "mnist")
#        print (model.conv1.weight)
        if (i==1):
            exit(0)    
        i+=1    

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


shelf = pymm.shelf("mnist",size_mb=1024,pmem_path="/mnt/pmem0/", force_new=True)
model = Net()
#shelf.torch_create_empty_model(model, "mnist")

##for name, param in model.named_parameters():
## setattr(shelf, name, torch.empty(param.size()))

#if args['cuda']:
#    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

shelf.save({
                    'epoch': np.zeros(1),
                    'model' : model,
                    'optimizer' : optimizer,
                    'loss': torch.empty(1) 
                }, header_name = "mnist", is_create_empty=True)
print("get_item_names")
print(shelf.get_item_names())
for epoch in range(1, args['epochs'] + 1):
    train(epoch)
#    test()



