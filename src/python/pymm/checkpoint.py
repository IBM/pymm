import pymmcore
from ctypes import *
from .check import paramcheck
import torch
import numpy as np
from collections import OrderedDict
import pickle

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
        type_name = type(data)
        
        # list
        if (type_name is list):   
           self.list_save(self, shelf, data, shelf_var_name, is_inplace)
           return

        # dict
        if (type_name is dict):    
           self.dict_save(self, shelf, data, shelf_var_name, is_inplace)
           return

        # ordered dict
        if (issubclass(type_name, OrderedDict)):    
           self.ordered_dict_save(self, shelf, data, shelf_var_name, is_inplace)
           return

        # in-place Numpy
        if (type_name is np.ndarray):
            if (shelf.__hasattr__(shelf_var_name)):
                self.numpy_save(self, shelf, data, shelf_var_name, is_inplace)
            else:     
                setattr(shelf, shelf_var_name, data)
            return

        # torch in-place
        if (self.is_type_torch(type_name)):
            if (shelf.__hasattr__(shelf_var_name)):
                self.torch_save(self, shelf, data, shelf_var_name, is_inplace)
            else:   
                setattr(shelf, shelf_var_name, data)
            return
 
        # basic shelf item
        if ((type_name is int) or (type_name is bool) or (type_name is str) or (type_name is bytes) or (type_name is float)): 
            setattr(shelf, shelf_var_name, data)
        else:
            print ("This type: " + str(type_name) + "is not support by the shelf, we save it on the shelf in byte format (using pickle)")
            setattr(shelf, shelf_var_name + "__pickle", pickle.dumps(data))
        

    ###############################################
    ###### save primitives #############
    ###############################################

     # list save 
    def list_save (self, shelf, list_items, shelf_var_name, is_inplace=True):
        for i in range(len(list_items)):
           self.save_manager(self, shelf, list_items[i], shelf_var_name + "__+list_" +  str(i), is_inplace)

     # Dict save
    def dict_save (self, shelf, data_dict, shelf_var_name, is_inplace=True):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], shelf_var_name + "__+dict_" +  str(name), is_inplace)

     # Ordered Dict save
    def ordered_dict_save (self, shelf, data_dict, shelf_var_name, is_inplace=True):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], shelf_var_name + "__+ordereddict_" +  str(name), is_inplace)
 
    # Save in-place Numpyarray
    def numpy_save(self, shelf, data, shelf_var_name, is_inplace=True):
            getattr(shelf, shelf_var_name)[:] = data

    # Save in-place Torch 
    def torch_save(self, shelf, data, shelf_var_name, is_inplace):
        with torch.no_grad():
            getattr(shelf, shelf_var_name).copy_(data)

###############################################
###############################################
##### Load ##########
###############################################
###############################################
    ###############################################
    ######### load checkpoint manager #################
    ###############################################

    def load_by_var_manager (self, shelf, target, shelf_var_name, is_inplace=True):
        type_name = type(target)

        if(self.is_type_torch_model(type_name)):
            return self.torch_load_model(self, shelf, target, shelf_var_name, is_inplace)

        # Torch Optim
        if (self.is_type_torch_optimizer(type_name)):
            return self.torch_load_optimizer(self, shelf, target, shelf_var_name, is_inplace)

        # list
        if (type_name is list):  
           return self.list_load(self, shelf, target, shelf_var_name + "__+list_", is_inplace)

        # dict
        if (type_name is dict):    
           return self.dict_load(self, shelf, target, shelf_var_name + "__+dict_", is_inplace)

        # get the item from the shelf
        return self.org_type(getattr(shelf, shelf_var_name))

    ###############################################
    ###### load primitives ##################
    ###############################################

    # load Torch Model
    def torch_load_model(self, shelf, model, shelf_var_name, is_inplace):
        for name, param in model.named_parameters():
            shelf_torch = self.load_by_var_manager(self, shelf, param, shelf_var_name + "__+model_#named_parameters_" + name, is_inplace)
            split_name = (name.rsplit('.', 1)) # split_name[0]: nmodel variable name, split_name[1]: bias /weight
            model_item = getattr(getattr(model, split_name[0].split(".")[0]), split_name[1])
            with torch.no_grad():
                 model_item.copy_(shelf_torch)
        return model

    # load Torch Optimizer 
    # in_place
    def torch_load_optimizer(self, shelf, opt, shelf_var_name, is_inplace):
        for k1, v1 in opt.state_dict().items():
             # Note 1: The optimizer.param_groups["param"][] does not contain the full tensor, 
             # but only an index to model.tensor, so we only save the index, and verify that
             # the optimizer.param_groups["param"][] is in the correct order during the load operation.
            if (k1 == "param_groups"):
                for i in range(len(v1)):
                    for k2, v2 in v1[i].items():
                        if (k2 == "params"):
                            for j in range(len(v2)):
                                # verification stage
                                if (v2[j] != getattr(shelf, shelf_var_name + "__+optimizer_#state_dict_param_groups__+list_" + str(i) + "__+dict_params__+list_" + str(j))):
                                    print ("Error in load optimizer, the optimizer.param_groups[\"params\"] are not correct")
                                    exit(0)
                                
                        else:    
                            opt.param_groups[i][k2] = self.load_by_var_manager(self, shelf, v2, shelf_var_name + "__+optimizer_#state_dict_param_groups__+list_" + str(i) + "__+dict_" + str(k2), is_inplace)

            # Note 2: The optimizer.state dictionary is empty, so we add the saved values
            if (k1 == "state"):
                list_shelf_items = shelf.get_item_names()
                shelf_val_2_add = []
                header = shelf_var_name + "__+optimizer_#state_dict_state__+dict_"
                for i in range(len(list_shelf_items)):
                    if (header in list_shelf_items[i]):
                       shelf_val_2_add.append(list_shelf_items[i].split(header, maxsplit=1)[1])
                for i in range(len(shelf_val_2_add)):
                    first_split = (shelf_val_2_add[i].split("__"))
                    second_split = first_split[1].split("+dict_")
                    opt.state[first_split[0]][second_split[1]] = self.org_type(getattr(shelf, header + shelf_val_2_add[i]))
        return 
    
    # load list 
    def list_load (self, shelf, list_items, shelf_var_name, is_inplace=True):
        # load inplace
        if (is_inplace):
            for i in range(len(list_items)):
                list_items[i] = self.load_by_var_manager(self, shelf, list_items[i], shelf_var_name + str(i), is_inplace)
            return  
        # return the value
        else:
            tmp_list = []
            for i in range(len(list_items)):
               tmp_list.append(self.load_by_var_manager(self, shelf, list_items[i], shelf_var_name + str(i)), is_inplace)
            return tmp_list





     # load dict
    def dict_load (self, shelf, data_dict, shelf_var_name, is_inplace=True):
        # load inplace
        if (is_inplace):
            for name in data_dict.keys():
               data_dict[name] = self.load_by_var_manager(self, shelf, data_dict[name], shelf_var_name + name, is_inplace)
            return    
        else: 
        # return the value
            tmp_dict = {}
            for name in data_dict.keys():
                tmp_dict[name] = self.load_by_var_manager(self, shelf, data_dict[name], shelf_var_name + name, is_inplace)
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
