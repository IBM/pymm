# Checkpoint 

The checkpoint method lets you put specific variables on the shelf and load tem later on. 

```
shelf.save (
                     data, 
                     header_name, 
                     is_inplace=True, 
                  )
```
## parameters
**data** - the data of the variable we place on the shelf, the variable could be nested, like a list of dicts.
Please refer to table 1 for the checkpointing types supported.
 
**header_name** - the header name for putting the data on the shelf. if the data_type is basic, then the header name is the shelf_name, but if it is complex, then each complex item will add an additional text to the header name (see column "add to header name" in table 1).

**in-place [default = True]** - This variable allows assigning variables in-place to existing variables on the shelf. It is supported in Torch and NumPy. If changed to False, then it would be out-of-place assigning. It uses setattr() under the hood. The shelf_item will be erased and the new data will be written. However, out-of-place is much slower than in-place.  You should use in_place=False when setting the item on the shelf for the first time.

### Table 1 : Types supported for checkpoiting.

| Type|Complex/ Basic  |add  to header name |
|----|---|---|
| dict | Complex | "__+dict_#keyname" |
| list | Complex | "__+list_ #iterate_num" |
| torch.model | Complex | "__+model_#named_parameters" |
| torch.optimizer | Complex | "__+optimizer_#param_groups" |
| torch | Basic (supported also in-place) | NO |
| NumPy | Basic (supported also in-place) | NO |
| Other basic PYMM supported types (int, float, sting..) | Basic (supported only out-of-place) | NO |


### Example:**
```
shelf.save( {   a = 1,   b = list["mylistzero", "mylistone"]}, header_name = example )
}
```
The shelf items are:
```
shelf.example__+dict_#a = 1
shelf.example__+dict_#b__+list_0 = "mylistzero"
shelf.example__+dict_#b__+list_1 = "mylistone"
```









