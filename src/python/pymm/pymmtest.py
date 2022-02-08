import sys
import numpy as np

def test_basic_writes():
    '''
    Test basic array writes
    '''
    import pymm
    import numpy as np
  
    # create new shelf (override any existing myShelf)
    #
    s = pymm.shelf('test_basic_writes',32,pmem_path='/mnt/pmem0',force_new=True)

    # create variable x on shelf (using shadow type)
    s.x = pymm.ndarray((1000,1000),dtype=np.uint8)

    if s.x.shape != (1000,1000):
        raise RuntimeError('demo: s.x.shape check failed')

    s.x[0] = 1
    if s.x[0] != 1:
        raise('Test 0: failure')

    if s.x[0][0] != 1:
        raise('Test 0: failure')

    s.x[0][0] = 2

    if s.x[0][0] != 2:
        raise('Test 0: failure')

    print('Test 0 OK!')


def test_write_operations():
    '''
    Test write operations
    '''
    import pymm
    import numpy as np
  
    # create new shelf (override any existing myShelf)
    #
    s = pymm.shelf('test_write_operations',32,pmem_path='/mnt/pmem0',force_new=True)

    # create variable x on shelf (using shadow type)
    s.x = pymm.ndarray((10,10),dtype=np.uint8)

    s.x.fill(1)

    # do not modify but make copies
    -s.x
    s.x+8 
    if s.x[0][0] != 1:
        raise RuntimeError('test failed unexpectedly')
    
    return s
    
    

#     def test_shelf_dtor():
#     import pymm
#     import gc
#     import sys
    
#     s = pymm.shelf('myShelf',32,pmem_path='/mnt/pmem0',force_new=True)
#     print(type(s))
#     s.x = pymm.ndarray((1000,1000),dtype=np.float)
#     s.y = pymm.ndarray((1000,1000),dtype=np.float)
#     print(s.items)
#     t = s.x
# #    u = s.x
# #    v = s.x

#     del s
# #    t.fill(8)
# #    print(t)
# #    del s
# #    gc.collect()
    

#     print('Shelf deleted explicitly')

#    print('refcnt(t)=', sys.getrefcount(t))
#    print('refcnt(u)=', sys.getrefcount(u))
#    print(t)
    
    
# def testX():
#     import pymm
#     import numpy as np
#     y = pymm.ndarray(shape=(4,4),dtype=np.uint8)
#     print("created ndarray subclass");
# #    hdr = pymm.pymmcore.ndarray_header(y);
#     print("hdr=", pymm.pymmcore.ndarray_header(y))
#     return None

# def test_shelf():
#     import pymm
#     import numpy as np

#     s = pymm.shelf('myShelf')
#     s.x = pymm.ndarray((8,8),dtype=np.uint8)

#     # implicity replace s.x (RHS constructor should succeed before existing s.x is erased)
# #    s.x = pymm.ndarray((3,3),dtype=np.uint8)
#     print(s.x)
#     return s

# def test_memoryresource():
#     import pymm
#     mr = pymm.MemoryResource('zzz', 1024)
#     nm = mr.create_named_memory('zimbar', 8)
#     print(nm.handle,hex(pymm.pymmcore.memoryview_addr(nm.buffer)))

# def test1():
#     print('test1 running...')
#     mr = MemoryResource()
#     return None

# def test2():
#     import pymm
#     import numpy as np
#     x = np.ndarray((8,8),dtype=np.uint8)
#     print(pymm.pymmcore.ndarray_header(x))
#     print(pymm.pymmcore.ndarray_header_size(x))

# def test3():
#     import pymm
#     import numpy as np
#     x = pymm.ndarray('foo',(4,4),dtype=np.uint8)
#     print(x.__name__)
#     print(x)

# def test_arrow0():
#     import pymm
#     import pyarrow as pa
#     data = [
#         pa.array([1, 2, 3, 4]),
#         pa.array(['foo', 'bar', 'baz', None]),
#         pa.array([True, None, False, True])
#     ]
    
#     raw_buffer = pymm.pymmcore.allocate_direct_memory(2048)
#     raw_buffer2 = pymm.pymmcore.allocate_direct_memory(2048)
#     print(hex(pymm.pymmcore.memoryview_addr(raw_buffer)))
#     buffer = pa.py_buffer(raw_buffer);
#     buffer2 = pa.py_buffer(raw_buffer2);
#     return pa.Array.from_buffers(pa.uint8(), 10, [buffer, buffer2])


###pointer, read_only_flag = a.__array_interface__['data']

# data = [
#     pa.array([1, 2, 3, 4]),
#     pa.array(['foo', 'bar', 'baz', None]),
#     pa.array([True, None, False, True])
# ]
# batch = pa.RecordBatch.from_arrays(data, ['f0', 'f1', 'f2'])
# table = pa.Table.from_batches([batch])

# raw_buffer = pymm.pymmcore.allocate_direct_memory(2048)
# raw_buffer2 = pymm.pymmcore.allocate_direct_memory(2048)
# print(hex(pymm.pymmcore.memoryview_addr(raw_buffer)))
# buffer = pa.py_buffer(raw_buffer);
# buffer2 = pa.py_buffer(raw_buffer2);
# return pa.Array.from_buffers(pa.uint8(), 10, [buffer, buffer2])
