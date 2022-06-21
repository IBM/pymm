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

        # save every two loops
        if (not (index % 2)):
            print ("save (iterator: i = " + str(index) + ")")
            shelf.save({
                    'epoch': np.array([epoch]),
                    'model' : model,
                    'optimizer' : optimizer,
                    'loss': loss.data 
            }, shelf_var_name = "mnist", is_inplace=True)
            save_model = {}
            for name, param in model.named_parameters():
                save_model[str(name)] = torch.clone(param) 
            save_loss_data = loss.data
            save_optimizer = []
            for i in range(len(optimizer.param_groups[0]["params"])):
                save_optimizer.append(torch.clone(optimizer.param_groups[0]["params"][i]))
         
        # load the saved value
        if (index % 2):
            print ("load (iterator: i = " + str(index-1) + "), we are in iterator: ", str(index))
#            optimizer.param_groups = shelf.load(optimizer, "mnist__+dict_optimizer")
            model = shelf.load(model, "mnist__+dict_model")
            epoch = shelf.load(epoch, "mnist__+dict_epoch")
            loss.data = shelf.load(loss.data, "mnist__+dict_loss")
            # check for correctness
            for name, param in model.named_parameters():
                if (not torch.equal(save_model[name], param)):
                    print ("Error: check that we load this item currectly - " + str(name))
#                    exit(0)
            if (not (save_loss_data == loss.data)): 
                print ("Error: check that we load this item currectly - loss.data")
                exit(0)
            for i in range(len(optimizer.param_groups[0]["params"])):
                if (not torch.equal(save_optimizer[i], optimizer.param_groups[0]["params"][i])):
                    print ("Error: check that we load this item currectly - optimizer.param_groups[0][params][" + str(i) + "]")
#                exit(0)
        print ("I am here " + str(index))
        if (index == 2):
            print ("Test pass")
            exit(0)    
        index+=1    

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
                }, shelf_var_name = "mnist", is_inplace=False)

# items on the shelf
print("get_item_names")
for epoch in range(1, args['epochs'] + 1):
    train(epoch)
#    test()



