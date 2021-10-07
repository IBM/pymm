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

def demo(force_new=True):
    '''
    Demonstration of pymm features
    '''
  
    # create new shelf (override any existing myShelf)
    #
    s = pymm.shelf('myShelf','/mnt/pmem0',1024, force_new=force_new)

    # create variable x on shelf (using shadow type)
    s.x = pymm.ndarray((1000,1000),dtype=np.float)

    if s.x.shape != (1000,1000):
        raise RuntimeError('demo: s.x.shape check failed')

    # perform in-place (on-shelf) operations
    s.x.fill(3)
    s.x += 2
    x_checksum = sum(s.x.tobytes()) # get checksum

    # write binary array data to file
    dfile = open("array.dat","wb")
    dfile.write(s.x.tobytes())
    dfile.close()

    # create new instance
    s.z = np.ndarray((1000,1000),dtype=np.float)

    # zero-copy read into instance from file
    with open("array.dat", "rb") as source:
        source.readinto(memoryview(s.z))
    z_checksum = sum(s.z.tobytes()) # get checksum
    if z_checksum != x_checksum:
        raise RuntimeError('data checksum mismatch')

    # this will create a persistent memory copy from RHS DRAM/volatile instance
    # the right hand side will be garbage collected
    from skimage import data, io

    s.i = data.camera()
    s.j = data.brick()

    s.blended = s.i + (0.5 * s.j)
    io.imshow(s.blended)
    io.show()

    # place a utf-16 string
    s.msg = pymm.string(b'\xff\xfe=\xd8\t\xde', encoding='utf-16')
    s.msg

    # remove objects from shelf
    for item in s.items:
        s.erase(item)
    
    return
