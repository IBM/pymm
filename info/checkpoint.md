
# Checkpoint 
Checkpoints allow you to store variables on the shelf and load them later.

## SAVE
```
shelf.save (
                     data, 
                     shelf_var_name, 
            )
```
### parameters
**data** - the data of the variable we place on the shelf, the variable could be nested, like a list of dicts.
Supported types can be saved on the shelf, other classes are saved as pickles. The types of checkpointing supported are listed in table 1.
 
**shelf_var_name** - the name of the shelf variable, the header for all the variables saved on the shelf. This header is used to retrieve shelf_variables during loading.   

Basic types are saved in separate variables, and the complex types are saved in many variables. For instance, Each key of dict will be placed in a separate variable. See examples.
The naming conventions added to shelf_var_name are shown in Table 1.

### Table 1 : Checkpoint - supported types and naming convention.
| Type|Complex/ Basic  |add to shelf_var_name |
|----|---|---|
| dict | Complex | "__+dict_#keytype#_#keyname#" |
| ordered dict | Complex | "__+dict_#keytype#_#keyname#" |
| list | Complex | "__+list_iterate_num" |
| torch | Basic PyMM Type | __#Torch# |
| NumPy | Basi PyMM Type | __#ndarray# |
| Other basic PYMM supported types (int, float, sting..) | Basic | __#type_name# |
| Not supported types | Save with pickle on the shelf | __#pickle#

* In Torch and Numpy, if you save to existing variable an in-place operation will be held.

### Example 1 - Save dict with list
```
>>> shelf.save(
          { 
              "a" : 1,
              "b" : ["mylistzero", "mylistone"]
          },
         shelf_var_name = "example" )
made int instance from copy 'example__+dict_#str#_#a#__#int#' on shelf
made str instance from copy 'example__+dict_#str#_#b#__+list_0__#str#' on shelf
made str instance from copy 'example__+dict_#str#_#b#__+list_1__#str#' on shelf
```

### Example 2 - Pytorch Save (model, optimizer, epoch, loss)
Save the state dict of the model and the optimizer.
The epoch in the shelf_var_name allows different version of the model
```
shelf.save({
                'epoch': epoch,
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'loss': loss.data
        }, shelf_var_name = "mnist" + str(epoch))
```

## Load 
```
shelf.load (
               shelf_var_name
            )
```
### parameters
**shelf_var_name** - shelf name the same as in the save. 

When loading variables from the shelf, we first collect all variables with the "shelf_var_name" header, sort them, then load them by name,
the saved variable is returned.


### Example 3 - Pytorch Load (model, optimizer, epoch, loss)

```
'''
Create an empty model, optimizer, and loss function
'''
model = Net()
optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
loss = F.nll_loss(output, target)

'''
Load the data from the shelf with the correct version.
'''
checkpoint_shelf = shelf.load("mnist" + str(epoch))

'''
Store the load_data in the variables
'''
model_shelf.load_state_dict(checkpoint_shelf['model'])
optimizer.load_state_dict(checkpoint_shelf['optimizer'])
epoch = checkpoint_shelf['epoch']
loss.data = checkpoint_shelf['loss']
```





