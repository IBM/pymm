# Checkpoint 

The checkpoint method lets you put specific variables on the shelf and load them later on. 

```
shelf.save (
                     data, 
                     shelf_var_name, 
                     is_inplace=True, 
                  )
```
## parameters
**data** - the data of the variable we place on the shelf, the variable could be nested, like a list of dicts.
Please refer to table 1 for the checkpointing types supported.
 
**shelf_var_name** - the shelf name identifying shelf to put data on. If the data_type is basic, then the shelf_var_name is the shelf_name, but if it is complex, then each complex item will add an additional suffix to the shelf_var_name (see column "add to shelf_var_name" in table 1).

**in-place [default = True]** - this variable allows updating variables in-place that already exist on the shelf. It is supported in Torch and NumPy. If set to False, then it would be out-of-place assignment; the existing shelved item is erased and the new data written. Note, out-of-place updates are potentially much slower than in-place.  You should use in_place=False when instantiating an item on the shelf for the first time.

### Table 1 : Types supported for checkpoiting.

| Type|Complex/ Basic  |add  to shelf_var_name |
|----|---|---|
| dict | Complex | "__+dict_keyname" |
| list | Complex | "__+list_iterate_num" |
| torch.model | Complex | "__+model_named_parameters" |
| torch.optimizer | Complex | "__+optimizer_param_groups" |
| torch | Basic (supported also in-place) | NO |
| NumPy | Basic (supported also in-place) | NO |
| Other basic PYMM supported types (int, float, sting..) | Basic (supported only out-of-place) | NO |


### Example:**
```
>>> s.save({   "a" : 1,   "b" : ["mylistzero", "mylistone"]}, shelf_var_name = "example" )
>>> s.get_item_names()
['example__+dict_a',  'example__+dict_b__+list_0',  'example__+dict_b__+list_1'  ]
```









