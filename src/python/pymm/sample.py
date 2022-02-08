# 
#    Copyright [2021] [IBM Corporation]
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
import pymm
import numpy as np

# create new shelf (override any existing myShelf)
#
s = pymm.shelf('myShelf','/mnt/pmem0',1024, force_new=True)
print("Created shelf OK.")

# create variable x on shelf (using shadow type)
s.x = np.random.uniform(low=-1.0, high=1.0, size=100000000)
print("Created array 'x' on shelf OK. Data at {}".format(s.x.addr))

# in-place random initialization (could be faster with vectorize + copy)
print(s.x)

# sort in-place
s.x.sort()
print("Sorted array 'x' on shelf OK. Data at {}".format(s.x.addr))
print(s.x)

print("Use s and s.x to access shelf...")



