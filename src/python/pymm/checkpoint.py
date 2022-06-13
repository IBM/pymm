import pymmcore
from ctypes import *
from .check import paramcheck
import torch
import numpy as np

class checkpoint():
       

##### Save ##########


# checkpoint manager
    def save_manager(self, shelf, data, header_name, is_inplace=True, is_create_empty=False):
        #torch model
        type_name = type(data)
        if(issubclass(type_name, torch.nn.Module)):
            self.torch_save_model(self, shelf, data, header_name, is_inplace, is_create_empty)
            return

        # Torch Optim   
        if ("torch.optim" in str(type_name)):
            self.torch_save_optimizer(self, shelf, data, header_name, is_inplace, is_create_empty)
            return

        # torch in-place
        if (is_inplace and "torch.nn.parameter" in str(type_name) or "torch.Tensor" in str(type_name)):
            self.torch_save(self, shelf, data, header_name, is_inplace, is_create_empty)
            return

        # list
        if (type_name is list):    
           self.list_save(self, shelf, data, header_name, is_inplace, is_create_empty)
           return

        # dict
        if (type_name is dict):    
           self.dict_save(self, shelf, data, header_name, is_inplace, is_create_empty)
           return

        # in-place Numpy
        if (is_inplace and type_name is np.ndarray):
            self.numpy_save(self, shelf, data, header_name, is_inplace, is_create_empty)

         # regular item
        setattr(shelf, header_name, data)


    # in-place Torch 
    def torch_save(self, shelf, data, header_name, is_inplace=True, is_create_empty=False):
        if (is_create_empty):
            setattr(shelf, header_name + "__+inplacetorch", torch.empty(data.size()))
        else:
            getattr(shelf, header_name + "__+inplacetorch").copy_(data)
 
#    def torch_load(self, header_name, torch_name):
#        with torch.no_grad():
#            torch_name.copy_(getattr(self, shelf_var))

    # Torch Model
    def torch_save_model(self, shelf, model, header_name, is_inplace=True, is_create_empty=False):
        for name, param in model.named_parameters():
            self.save_manager(self, shelf, param, header_name + "__+model_#" + name , is_inplace, is_create_empty)



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


    # Torch Optimizer 
    def torch_save_optimizer(self, shelf, opt, header_name, is_inplace, is_create_empty):
            self.save_manager(self, shelf, opt.param_groups, header_name + "__+optimizer_#param_groups", is_inplace, is_create_empty)

     # list 
    def list_save (self, shelf, list_items, header_name, is_inplace=True, is_create_empty=False):
        for i in range(len(list_items)):
           self.save_manager(self, shelf, list_items[i], header_name + "__+list_" +  str(i), is_inplace, is_create_empty)

     # list 
    def dict_save (self, shelf, data_dict, header_name, is_inplace=True, is_create_empty=False):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], header_name + "__+dict_" +  name, is_inplace, is_create_empty)

    # in-place Numpyarray
    def numpy_save(self, shelf, data, header_name, is_inplace=True, is_create_empty=False):
         if (is_create_empty):
            setattr(self, header_name + "__inplacenumpy", data)
         else:
            getattr(self, header_name + "__inplacenumpy")[:] = data


