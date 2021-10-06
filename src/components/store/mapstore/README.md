# Mapstore

Mapstore is an in-memory store that used std C++ unordered_set and DRAM.  Hence, there is no
crash consistency or persistence.

## Backing store

If you wish to use a file to provide mmap'ed memory, use the following environment variable:

```bash
MAPSTORE_BACKING_STORE_DIR=/tmp 
```

With this parameter, mapstore will create a file for each pool (mapstore_backing_<poolname>.dat)
and then mmap onto it. This is useful when you want to use larger than available memory; the
system will page the file.


