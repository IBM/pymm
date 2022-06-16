import pymmcore
from ctypes import *
from .check import paramcheck
import torch
import numpy as np

class checkpoint():
       

##### Save ##########


    ###############################################
    #########  checkpoint manager #################
    ###############################################
    def save_manager(self, shelf, data, header_name, is_inplace=True):
        #torch model
        type_name = type(data)
        if(self.is_type_torch_model(type_name)):
            self.torch_save_model(self, shelf, data, header_name, is_inplace)
            return

        # Torch Optim   
        if (self.is_type_torch_optimizer(type_name)):
            self.torch_save_optimizer(self, shelf, data, header_name, is_inplace)
            return

        # torch in-place
        if (is_inplace and self.is_type_torch(type_name)):  
            self.torch_save(self, shelf, data, header_name, is_inplace)
            return

        # list
        if (type_name is list):    
           self.list_save(self, shelf, data, header_name, is_inplace)
           return

        # dict
        if (type_name is dict):    
           self.dict_save(self, shelf, data, header_name, is_inplace)
           return

        # in-place Numpy
        if (is_inplace and type_name is np.ndarray):
            self.numpy_save(self, shelf, data, header_name, is_inplace)
            return
         # regular item
        setattr(shelf, header_name, data)


    ###############################################
    ###### checkpoint primitives ##################
    ###############################################
    # Save in-place Torch 
    def torch_save(self, shelf, data, header_name, is_inplace=True):
        with torch.no_grad():
            getattr(shelf, header_name).copy_(data)
 
#    def torch_load(self, header_name, torch_name):
#        with torch.no_grad():
#            torch_name.copy_(getattr(self, shelf_var))

    # Save Torch Model
    def torch_save_model(self, shelf, model, header_name, is_inplace=True):
        for name, param in model.named_parameters():
            self.save_manager(self, shelf, param, header_name + "__+model_#named_parameters" + name , is_inplace)



#    def torch_load_model(self, 
#    def torch_load_model(self, model, header_name):
#        for shelf_var in self.get_item_names():
#            if(shelf_var.startswith(header_name)):
#                 name = shelf_var.lstrip(header_name + "__")
#                 split_name = (name.rsplit('.', 1)) # split_name[0]: nmodel variable name, split_name[1]: bias /weight
#                 model_dist = getattr(getattr(model, split_name[0].split(".")[0]), split_name[1])
#                 shelf_src = getattr(self, shelf_var)
#                 with torch.no_grad():
#                     model_dist.copy_(shelf_src)


    # Save Torch Optimizer 
    def torch_save_optimizer(self, shelf, opt, header_name, is_inplace):
            self.save_manager(self, shelf, opt.param_groups, header_name + "__+optimizer_#param_groups", is_inplace)

     # list save 
    def list_save (self, shelf, list_items, header_name, is_inplace=True):
        for i in range(len(list_items)):
           self.save_manager(self, shelf, list_items[i], header_name + "__+list_" +  str(i), is_inplace)

     # Dict save
    def dict_save (self, shelf, data_dict, header_name, is_inplace=True):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], header_name + "__+dict_" +  name, is_inplace)

    # Save in-place Numpyarray
    def numpy_save(self, shelf, data, header_name, is_inplace=True):
            getattr(shelf, header_name)[:] = data


    
    ###############################################
    ###### check types  ###########################
    ###############################################

    def is_type_torch(type_name):
         return ("torch.nn.parameter" in str(type_name) or "torch.Tensor" in str(type_name))

    def is_type_torch_model(type_name):
         return (issubclass(type_name, torch.nn.Module))

    def is_type_torch_optimizer(type_name):
        return("torch.optim" in str(type_name))
