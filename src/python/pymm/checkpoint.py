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
    def save_manager(self, shelf, data, shelf_var_name):
        type_name = type(data)
        
        # list
        if (type_name is list):   
           self.list_save(self, shelf, data, shelf_var_name)
           return

        # dict
        if (type_name is dict):    
           self.dict_save(self, shelf, data, shelf_var_name)
           return

        # ordered dict
        if (issubclass(type_name, OrderedDict)):    
           self.ordered_dict_save(self, shelf, data, shelf_var_name)
           return

        # in-place Numpy
        if (type_name is np.ndarray):
            shelf_var_name = shelf_var_name + "__#" + type_name.__name__ + "#"
            if (shelf.__hasattr__(shelf_var_name)):
                self.numpy_save(self, shelf, data, shelf_var_name)
            else:     
                setattr(shelf, shelf_var_name, data)
            return

        # torch in-place
        if (self.is_type_torch(type_name)):
            shelf_var_name = shelf_var_name + "__#" + type_name.__name__ + "#"
            if (shelf.__hasattr__(shelf_var_name)):
                self.torch_save(self, shelf, data, shelf_var_name)
            else:   
                setattr(shelf, shelf_var_name, data)
            return
 
        # basic shelf item
        if ((type_name is int) or (type_name is bool) or (type_name is str) or (type_name is bytes) or (type_name is float)):
            shelf_var_name = shelf_var_name + "__#" + type_name.__name__ +"#"
            setattr(shelf, shelf_var_name, data)
        else:
            setattr(shelf, shelf_var_name + "__#pickle#", pickle.dumps(data))
        

    ###############################################
    ###### save primitives #############
    ###############################################

     # list save 
    def list_save (self, shelf, list_items, shelf_var_name):
        for i in range(len(list_items)):
           self.save_manager(self, shelf, list_items[i], shelf_var_name + "__+list_" + str(i))

     # Dict save
    def dict_save (self, shelf, data_dict, shelf_var_name):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], shelf_var_name + "__+dict_#" +  type(name).__name__ + "#_#"  + str(name) +"#" )

     # Ordered Dict save
    def ordered_dict_save (self, shelf, data_dict, shelf_var_name):
        for name in data_dict.keys():
           self.save_manager(self, shelf, data_dict[name], shelf_var_name + "__+ordereddict_#" + type(name).__name__ + "#_#"  + str(name) +"#")
 
    # Save in-place Numpyarray
    def numpy_save(self, shelf, data, shelf_var_name):
            getattr(shelf, shelf_var_name)[:] = data

    # Save in-place Torch 
    def torch_save(self, shelf, data, shelf_var_name):
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

    def load_manager (self, shelf, shelf_var_name):
        
        all_items = shelf.get_item_names()
        '''
        Add only items withi the header shelf_var_name
        '''
        items = []
        for item in all_items:
            if (item.startswith(shelf_var_name)):
                items.append(item)

        '''
        No items for this shelf_var_name
        '''
        if (len(items) == 0):
            print ("no items to load")
            return None

        '''
        One type - a basic type
        '''
        if (len(items) == 1) and ("+" not in items[0]):
            chop_item = items[0].split(shelf_var_name + "__")[1]
            return self.load_basic_type(self, shelf, items[0], [chop_item], 0)
         
        items.sort()
        chop_first_item = items[0].split(shelf_var_name + "__")[1]
        ret_val = self.load_initial_type(chop_first_item)

        for item in items:
            chop_item = item.split(shelf_var_name + "__")[1]
            item_split = chop_item.split("__")
            index = 0
            ret_val = self.load_item(self, shelf, item, item_split, index, ret_val)
        return ret_val        

    def load_initial_type (item):
        if item.startswith("+dict"):
            return {}
        if item.startswith("+ordereddict"):
            return OrderedDict() 
        if item.startswith("+list"):
            return []
        # Not supported type
        print ("There is something wrong with initialization, the first item is: " +  item)
        exit(0)


    def load_item (self, shelf, item_name, item_split, index, ret_val):    
        if item_split[index].startswith("+"):
            '''
            load complex type
            '''
            return self.load_complex_type(self, shelf, item_name, item_split, index, ret_val)
        if item_split[index].startswith("#"):
            '''
            load basic type
            '''
            return self.load_basic_type(self, shelf, item_name, item_split, index)
        '''
        Not supported type
        '''
        print ("something is wrong with loading " +  item)
        exit(0)   
 

    def load_basic_type (self, shelf, item_name, item_split, index):
        shelf_type = item_split[index].split("#")[1]
        return self.cast_var(shelf_type, getattr(shelf, item_name))

    def load_complex_type (self, shelf, item_name, item_split, index, ret_val):
        if item_split[index].startswith("+dict"):
            '''
            load dict 
            '''
            if (ret_val is None):
                ret_val = {}
            return self.load_dict(self, shelf, item_name, item_split, index, ret_val)
        if item_split[index].startswith("+ordereddict"):
            '''
            load ordereddict
            '''
            if (ret_val is None):
                ret_val = OrderedDict()
            return self.load_ordereddict(self, shelf, item_name, item_split, index, ret_val)

        if item_split[index].startswith("+list"):
            '''
            load list
            '''
            if (ret_val is None):
                ret_val = []
            return self.load_list(self, shelf, item_name, item_split, index, ret_val)
        '''
        Not supported type
        '''
        print ("There is something wrong with loading " +  item_name + \
                " --- we are at item_split " + item_split[index])
        exit(0)

 
      
    def load_dict (self, shelf, item_name, item_split, index, ret_val):
        split_item = item_split[index].split("+dict_")[1].split("#")
        key_name = split_item[3]
        key_type = split_item[1]
        index += 1
        if self.cast_var(key_type, key_name) not in ret_val: 
            ret_val[self.cast_var(key_type, key_name)] = None
        ret_val[self.cast_var(key_type, key_name)] = \
                self.load_item(self, shelf, item_name, item_split, index, ret_val[self.cast_var(key_type, key_name)])   
        return ret_val        

    def load_ordereddict (self, shelf, item_name, item_split, index, ret_val):
        split_item = item_split[index].split("+ordereddict_")[1].split("#")
        key_name = split_item[3]
        key_type = split_item[1]
        index += 1
        if self.cast_var(key_type, key_name) not in ret_val:  
            ret_val[self.cast_var(key_type, key_name)] = None
        ret_val[self.cast_var(key_type, key_name)] = \
                self.load_item(self, shelf, item_name, item_split, index, ret_val[self.cast_var(key_type, key_name)])   
        return ret_val        

    def load_list (self, shelf, item_name, item_split, index, ret_val):
        list_idx = int(item_split[index].split("_")[1])
        if (len(ret_val) <= list_idx):
            # we append one item
            assert (len(ret_val) == list_idx)
            ret_val.append(None)
        index += 1    
        ret_val[list_idx] = \
                self.load_item(self, shelf, item_name, item_split, index, ret_val[list_idx])   
        return ret_val        






    ###############################################
    ###### load primitives ##################
    ###############################################


    def cast_var (type_var, var):
        if (type_var  == "str"):
            return str(var)
        if (type_var == "int"):
            return int(var)
        if (type_var == "float"):
            return float(var)
        if (type_var == "bytes"):
            return bytes(var)
        if (type_var == "bool"):
            return bool(var)
        if (type_var == "ndarray"):
            return np.array(var)
        if (type_var == "Tensor"):
            return torch.clone(var)
        if (type_var == "pickle"):
            return pickle.loads(bytes(var))
         
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
