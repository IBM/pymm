#Import Libraries
'''
In this test, We save and load the MNISM model before and during training, 
in addition to the model we save the optimizer the loass and the epoch numebr.
We verify the the saved parametres are identical to the running parameters.
'''

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


def compare_item (item1, item2, name):
    if (type(item1) != type(item2)):
        print ("Mismatch found type error in " + name + ": " + str(type(item1)) + "/" + str(type(item2)))
        exit(0)
    if (item1 != item2):
        print ("Mismatch found type error in " + name + ": " + str(item1) + "/" + str(item2))
        exit(0)
        


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
           print('Mismatch found at', key_item_1[0])
           exit(0)
    if models_differ == 0:
        print('Models match perfectly! :)')

def compare_optimizer(opt_1, opt_2):
    opt_differ = 0
    for key_item_1, key_item_2 in zip(opt_1.state_dict().items(), opt_2.state_dict().items()):
#        print (key_item_1[0])
        if (type(key_item_1[1]) == dict):
            compare_dict(key_item_1[1], key_item_2[1])
            continue
        if (type(key_item_1[1]) == list):
            compare_list(key_item_1[1], key_item_2[1])
            continue
        else:
            if type(key_item_1).__name__ is "Tensor":
                if torch.equal(key_item_1[1], key_item_2[1]):
                    continue
                else:
                   print('Mismatch found at', key_item_1[0])
                   exit(0)
            if (key_item_1[1] == key_item_2[1]):
                pass
            else: 
                print('Mismatch found at', key_item_1[0])
                exit(0)
    print('Optimizer match perfectly! :)')


def compare_dict (dict1, dict2):
    for key1 in dict1.keys(): 
#        print (key1)
        if (type(dict1[key1]) is dict):
            compare_dict(dict1[key1], dict2[key1])
            continue
        if (type(dict1[key1]) is list):
            compare_list(dict1[key1], dict2[key1])
            continue
        else:
            if type(dict1[key1]).__name__ is "Tensor":
                if torch.equal(dict1[key1], dict2[key1]):
                    continue
                else:
                   print('Mismatch found at {} --- ', key1)
                   exit(0)
            if (dict1[key1] == dict2[key1]):
                pass
            else: 
                print("Mismatch found at--- " + key1 + "--- "  + str(dict1[key1]) + "/"\
                        + str(dict2[key1])) 
                exit(0)

def compare_list (list1, list2):
    for idx in range(len(list1)):   
 #       print ("list " + str(idx))
        if (type(list1[idx]) == dict):
            compare_dict(list1[idx], list2[idx])
            continue
        if (type(list1[idx]) == list):
            compare_list(list1[idx], list2[idx])
            continue
        else:
            if type(list1[idx]).__name__ is "torch.Tensor":
                if torch.equal(list1[idx], list2[idx]):
                    continue
                else:
                   print('Mismatch found at [] --- ', list1[idx])
                   exit(0)
            if (list1[idx] == list2[idx]):
                pass
            else: 
                print('Mismatch found  at [] --- ', list1[idx])
                exit(0)

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
    global model
    model.train()
    index = 0
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
        index+=1    

        shelf.save({
                'epoch': epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss': loss.data 
        }, shelf_var_name = "mnist" + str(epoch))
        model_shelf = Net()
        optimizer_shelf = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
        checkpoint_shelf = shelf.load("mnist" + str(epoch))
        model_shelf.load_state_dict(checkpoint_shelf['model'])
        optimizer_shelf.load_state_dict(checkpoint_shelf['optimizer'])
        
        compare_models(model, model_shelf)
        compare_optimizer(optimizer, optimizer_shelf)
        compare_item(epoch, checkpoint_shelf['epoch'], "epoch")
        compare_item(loss, checkpoint_shelf['loss'], "loss")
        print ("Test pass!!!!")
        exit(0)

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

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])


shelf.save({
                    'epoch': 0,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'loss': torch.empty(1) 
                }, shelf_var_name = "mnist" + str(0))

model_shelf = Net()
optimizer_shelf = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
checkpoint_shelf = shelf.load("mnist" + str(0))
optimizer_shelf.load_state_dict(checkpoint_shelf['optimizer'])
compare_optimizer(optimizer, optimizer_shelf)
model_shelf.load_state_dict(checkpoint_shelf['model'])
compare_models(model, model_shelf)
compare_optimizer(optimizer, optimizer)
# items on the shelf
for epoch in range(1, args['epochs'] + 1):
    train(epoch)
#    test()



