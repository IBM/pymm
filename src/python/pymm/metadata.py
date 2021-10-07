import pymmcore
from ctypes import *
from .check import paramcheck

class MetaHeader(Structure):
    _fields_ = [("magic", c_uint32),
                ("txbits", c_uint32),
                ("version", c_uint32),
                ("type", c_uint32),
                ("subtype", c_uint32),
                ("refcnt", c_uint32),
    ]


# corresponds to core/metadata.h C definition

def construct_header(type=0, subtype=0, txbits=0, version=0):
    hdr = MetaHeader()
    hdr.magic = HeaderMagic
    hdr.txbits = txbits
    hdr.version = version
    hdr.type = type
    hdr.refcnt = 0
    return hdr

def construct_header_on_buffer(buffer, type=0, subtype=0, txbits=0, version=0):
    hdr = MetaHeader.from_buffer(buffer)
    hdr.magic = HeaderMagic
    hdr.txbits = txbits
    hdr.version = version
    hdr.type = type
    hdr.refcnt = 0
    return hdr

@paramcheck(types=[memoryview])
def init_header_from_buffer(buffer: memoryview):
    hdr = MetaHeader.from_buffer(buffer)
    hdr.magic = HeaderMagic
    hdr.txbits = 0
    hdr.version = 0
    hdr.refcnt = 0
    return hdr

@paramcheck(types=[memoryview])
def construct_header_from_buffer(buffer: memoryview):
    hdr = MetaHeader.from_buffer(buffer)
    if hdr.magic != HeaderMagic:
        raise RuntimeError('bad magic: {} construct header from buffer'.format(hex(hdr.magic)))
    return hdr

@paramcheck(types=[memoryview])
def metadata_check_header(buffer: memoryview):
    hdr = MetaHeader.from_buffer(buffer)
    return hdr.magic == HeaderMagic

@paramcheck(types=[MetaHeader])
def metadata_set_dirty_tx_bit(metadata_header):
    '''
    Set and persist the TXBIT_DIRTY in the header
    '''
    assert not(metadata_header.txbits & TXBIT_DIRTY)
    metadata_header.txbits |= TXBIT_DIRTY
    pymmcore.persist(memoryview(metadata_header))

@paramcheck(types=[MetaHeader])    
def metadata_clear_dirty_tx_bit(metadata_header):
    '''
    Clear and persist the TXBIT_DIRTY in the header
    '''
    assert metadata_header.txbits & TXBIT_DIRTY
    metadata_header.version += 1
    metadata_header.txbits &= ~TXBIT_DIRTY
    pymmcore.persist(memoryview(metadata_header))

@paramcheck(types=[MetaHeader])    
def metadata_check_dirty_tx_bit(metadata_header):
    '''
    Clear and persist the TXBIT_DIRTY in the header
    '''
    return metadata_header.txbits & TXBIT_DIRTY


@paramcheck(types=[MetaHeader, int])
def metadata_set_tx_bit(metadata_header, mask):
    '''
    Set and persist the mask bits in the header
    '''
    assert not(metadata_header.txbits & mask)
    metadata_header.txbits |= mask
    pymmcore.persist(memoryview(metadata_header))

@paramcheck(types=[MetaHeader, int])    
def metadata_clear_tx_bit(metadata_header, mask):
    '''
    Clear and persist the mask bits in the header
    '''
    assert metadata_header.txbits & mask
    metadata_header.txbits &= ~mask
    pymmcore.persist(memoryview(metadata_header))

@paramcheck(types=[MetaHeader, int])    
def metadata_check_tx_bit(metadata_header, mask):
    '''
    Check if mask bits in the header are set
    '''
    return (metadata_header.txbits & mask) == mask

    
    

HeaderSize = 24
HeaderMagic = int(0xCAF0)

DataType_Unknown       = int(0)
DataType_Opaque        = int(1)
DataType_String        = int(2)
DataType_NumberFloat   = int(3)
DataType_NumberInteger = int(4)
DataType_Bytes         = int(5)
DataType_NumPyArray    = int(10)
DataType_TorchTensor   = int(11)
DataType_DLTensor      = int(12)
DataType_LinkedList    = int(23)

DataSubType_None   = int(0)
DataSubType_Ascii  = int(10)
DataSubType_Utf8   = int(11)
DataSubType_Utf16  = int(12)
DataSubType_Latin1 = int(13)

TXBIT_DIRTY       = int(1 << 0) # changes could exist that have not yet been flushed
TXBIT_MULTIVAR    = int(1 << 1) # variable is part of a multi-variable transaction

