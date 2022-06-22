import pymmcore
from ctypes import *
from .check import paramcheck
import torch
import numpy as np

class checkpoint():
       

###############################################
###############################################
##### Save ##########
###############################################
###############################################
    ###############################################
    ######### save checkpoint manager #################
    ###############################################
    def save_manager(self, shelf, data, shelf_var_name, is_inplace=True):
        #torch model
        type_name = type(data)
        if(self.is_type_torch_model(type_name)):
            self.torch_save_model(self, shelf, data, shelf_var_name, is_inplace)
            return

        # Torch Optim   
        if (self.is_type_torch_optimizer(type_name)):
            self.torch_save_optimizer(self, shelf, data, shelf_var_name, is_inplace)
            return

        # torch in-place
        if (is_inplace and self.is_type_torch(type_name)):  
            self.torch_save(self, shelf, data, shelf_var_name, is_inplace)
            return

        # list
        if (type_name is list):    
           self.list_save(self, shelf, data, shelf_var_name, is_inplace)
           return

        # dict
        if (type_name is dict):    
           self.dict_save(self, shelf, data, shelf_var_name, is_inplace)
           return

        # in-place Numpy
        if (is_inplace and type_name is np.ndarray):
            self.numpy_save(self, shelf, data, shelf_var_name, is_inplace)
            return
         # regular item

        setattr(shelf, shelf_var_name, data)


    ###############################################
    ###### save primitives #############
    ###############################################
    # Save in-place Torch 
    def torch_save(self, shelf, data, shelf_var_name, is_inplace):
        with torch.no_grad():
            getattr(shelf, shelf_var_name).copy_(data)
 
    # Save Torch Model
    def torch_save_model(self, shelf, model, shelf_var_name, is_inplace):
        for name, param in model.named_parameters():
            self.save_manager(self, shelf, param, shelf_var_name + "__+model_#named_parameters_" + name , is_inplace)


    # Save Torch Optimizer 
    def torch_save_optimizer(self, shelf, opt, shelf_var_name, is_inplace):
            for k,v in opt.state_dict().items():
                if (k != "state"):
                    self.save_manager(self, shelf, v, shelf_var_name + "__+optimizer_#state_dict_" + str(k), is_inplace)

     # list save 
    def list_save (self, shelf, list_items, shelf_var_name, is_inplace=True):
        for i in range(len(list_items)):
           self.save_manager(self, shelf, list_items[i], shelf_var_name + "__+list_" +  str(i), is_inplace)

     # Dict save
    def dict_save (self, shelf, data_dict, shelf_var_name, is_inplace=True):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], shelf_var_name + "__+dict_" +  name, is_inplace)

    # Save in-place Numpyarray
    def numpy_save(self, shelf, data, shelf_var_name, is_inplace=True):
            getattr(shelf, shelf_var_name)[:] = data



###############################################
###############################################
##### Load ##########
###############################################
###############################################
    ###############################################
    ######### load checkpoint manager #################
    ###############################################

    def load_by_var_manager (self, shelf, target, shelf_var_name):
        
        type_name = type(target)
        if(self.is_type_torch_model(type_name)):
            return self.torch_load_model(self, shelf, target, shelf_var_name)

        # Torch Optim   
        if (self.is_type_torch_optimizer(type_name)):
            return self.torch_load_optimizer(self, shelf, target, shelf_var_name)

        # list
        if (type_name is list):   
           return self.list_load(self, shelf, target, shelf_var_name + "__+list_")

        # dict
        if (type_name is dict):    
           return self.dict_load(self, shelf, target, shelf_var_name + "__+dict_")

        # get the item from the shelf
        return self.org_type(getattr(shelf, shelf_var_name))

    ###############################################
    ###### load primitives ##################
    ###############################################

    # load Torch Model
    def torch_load_model(self, shelf, model, shelf_var_name):
        for name, param in model.named_parameters():
            shelf_torch = self.load_by_var_manager(self, shelf, param, shelf_var_name + "__+model_#named_parameters_" + name)
            split_name = (name.rsplit('.', 1)) # split_name[0]: nmodel variable name, split_name[1]: bias /weight
            model_item = getattr(getattr(model, split_name[0].split(".")[0]), split_name[1])
            with torch.no_grad():
                 model_item.copy_(shelf_torch)
        return model

    # load Torch Optimizer 
    def torch_load_optimizer(self, shelf, opt, shelf_var_name):
        return self.load_by_var_manager(self, shelf, opt.param_groups, shelf_var_name + "__+optimizer_#param_groups")

     # load list 
    def list_load (self, shelf, list_items, shelf_var_name):
        tmp_list = []
        for i in range(len(list_items)):
           tmp_list.append(self.load_by_var_manager(self, shelf, list_items[i], shelf_var_name + str(i)))
        return tmp_list

     # load dict
    def dict_load (self, shelf, data_dict, shelf_var_name):
        tmp_dict = {}
        for name in data_dict.keys():
           tmp_dict[name] = self.load_by_var_manager(self, shelf, data_dict[name], shelf_var_name + name)
        return tmp_dict   

    def org_type (shelf_var):
        if ("pymm.integer_number" in str(type(shelf_var))):
            return int(shelf_var)
        if ("pymm.float_number" in str(type(shelf_var))):
            return float(shelf_var)
        if ("pymm.bytes" in str(type(shelf_var))):
            return bytes(shelf_var)
        if ("pymm.string" in str(type(shelf_var))):
            return str(shelf_var)
        if ("pymm.ndarray" in str(type(shelf_var))):
            return np.array(shelf_var)
        if ("pymm.torch_tensor" in str(type(shelf_var))):
            return torch.clone(shelf_var)
         
###############################################
###############################################
###### help function  ###########################
###############################################
###############################################
    ###############################################
    ###### check types  ###########################
    ###############################################

    def is_type_torch(type_name):
         return ("torch.nn.parameter" in str(type_name) or "torch.Tensor" in str(type_name))

    def is_type_torch_model(type_name):
         return (issubclass(type_name, torch.nn.Module))

    def is_type_torch_optimizer(type_name):
        return("torch.optim" in str(type_name))
