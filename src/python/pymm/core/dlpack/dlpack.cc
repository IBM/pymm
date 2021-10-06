/*
  Copyright [2017-2021] [IBM Corporation]
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pymmcore_ARRAY_API

#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/dump_utils.h>

#include <Python.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#include <numpy/arrayobject.h>
#pragma GCC diagnostic pop

#include "metadata.h"
#include "dlpack.h"

typedef int tvm_index_t;
typedef unsigned char bytes;

#if 0
static size_t dlpack_get_data_size(DLDataType dtype, int ndim) {
   size_t size = 1;
   for (tvm_index_t i = 0; i < ndim; ++i) {
     size *= t->shape[i];
   }
   size *= (t->dtype.bits * t->dtype.lanes + 7) / 8;
  return size;
}

static size_t dlpack_get_DLTensor_size(int ndim) {
  size_t nbytes = sizeof(DLTensor);
  nbytes += sizeof(int64_t) * ndim * 2;
  nbytes += dlpack_get_data_size
  return 0;
}
#endif



PyObject * pymmcore_dlpack_construct_meta(PyObject * self,
                                          PyObject * args,
                                          PyObject * kwargs)
{
  static const char *kwlist[] = {"dtypedescr",
                                 "shape",
                                 "strides",
                                 NULL};

  PyObject * dtypedescr_obj = nullptr;
  PyObject * shape_obj = nullptr;
  PyObject * strides_obj = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "OOO",
                                    const_cast<char**>(kwlist),
                                    &dtypedescr_obj,
                                    &shape_obj,
                                    &strides_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyArray_DescrCheck(dtypedescr_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad type of dtypedescr parameter");
    return NULL;
  }

  /* convert from numpy type to dlpack type */
  DLDataType ddt;
  
  auto dtypedescr = reinterpret_cast<PyArray_Descr*>(dtypedescr_obj);
  switch(dtypedescr->kind) {
  case 'i': // signed int
    ddt.code = kDLInt;
    break;
  case 'u': // unsigned int
    ddt.code = kDLUInt;
    break;
  case 'f': // float
    ddt.code = kDLFloat;
    break;
  case 'c': // complex
    ddt.code = kDLComplex;
  default:
    PyErr_SetString(PyExc_RuntimeError,"unsupport type");
    return NULL;
  }
  ddt.lanes = 1; /* no vectorization */
  ddt.bits = dtypedescr->elsize * 8;

  PLOG("dtypedescr.type_num = %d", dtypedescr->type_num);
  PLOG("ddt.code = %u", ddt.code);

  /* handle shape */
  std::vector<int64_t> c_shape;
  if (PyList_Check(shape_obj)) {
    for(Py_ssize_t idx = 0; idx < PyList_Size(shape_obj); idx++) {
      auto element = PyList_GetItem(shape_obj, idx);
      unsigned long n = PyLong_AsUnsignedLong(element);
      if (n == static_cast<unsigned long>(-1L)) {
        PyErr_SetString(PyExc_RuntimeError,"bad shape element");
        return NULL;
      }
      c_shape.push_back(n);
    }
  }
  else if (PyTuple_Check(shape_obj)) {
    for(Py_ssize_t idx = 0; idx < PyTuple_Size(shape_obj); idx++) {
      auto element = PyTuple_GetItem(shape_obj, idx);
      unsigned long n = PyLong_AsUnsignedLong(element);
      if (n == static_cast<unsigned long>(-1L)) {
        PyErr_SetString(PyExc_RuntimeError,"bad shape element");
        return NULL;
      }
      c_shape.push_back(n);
    }
  }
  else {
    PyErr_SetString(PyExc_RuntimeError,"shape should be list or tuple");
    return NULL;
  }

  /* handle strides */
  std::vector<int64_t> c_strides;
  if (Py_None == strides_obj) {
  }
  else if (PyList_Check(strides_obj)) {
    for(Py_ssize_t idx = 0; idx < PyList_Size(strides_obj); idx++) {
      auto element = PyList_GetItem(strides_obj, idx);
      unsigned long n = PyLong_AsUnsignedLong(element);
      if (n == static_cast<unsigned long>(-1L)) {
        PyErr_SetString(PyExc_RuntimeError,"bad shape element");
        return NULL;
      }
      c_strides.push_back(n);
    }
  }
  else if (PyTuple_Check(strides_obj)) {
    for(Py_ssize_t idx = 0; idx < PyTuple_Size(strides_obj); idx++) {
      auto element = PyTuple_GetItem(strides_obj, idx);
      unsigned long n = PyLong_AsUnsignedLong(element);
      if (n == static_cast<unsigned long>(-1L)) {
        PyErr_SetString(PyExc_RuntimeError,"bad shape element");
        return NULL;
      }
      c_strides.push_back(n);
    }
  }
  else {
    PyErr_SetString(PyExc_RuntimeError,"shape should be list or tuple");
    return NULL;
  }

  if(!c_strides.empty() && !c_shape.empty()) {
    if(c_strides.size() != c_shape.size()) {
      PyErr_SetString(PyExc_RuntimeError,"stride and shape count mismatch");
      return NULL;
    }
  }

  size_t data_size = 1;
  for(auto d: c_shape)
    data_size *= d;
  
  size_t ndim = c_shape.size();
  PLOG("ndim=%lu", ndim);
  MetaHeader metadata_header;
  metadata_header.magic = HeaderMagic;
  metadata_header.version = 0;
  metadata_header.txbits = 0;
  metadata_header.type = DataType_DLTensor;
  metadata_header.subtype = DataSubType_None;
  metadata_header.refcnt = 0;

  /* pointers data, shape, strides set to nullptr */
  DLTensor tensor;
  tensor.data = reinterpret_cast<int64_t*>(-1UL);
  tensor.device = {kDLCPU, 0};
  tensor.ndim = boost::numeric_cast<int>(ndim);
  tensor.dtype = ddt;
  tensor.shape = reinterpret_cast<int64_t*>(-1UL);
  tensor.strides = c_strides.empty() ? nullptr : reinterpret_cast<int64_t*>(-1UL);
  tensor.byte_offset = 0;
  
  /* now construct the header - copy for moment */
  std::stringstream hdr;
  hdr.write(reinterpret_cast<const char*>(&metadata_header), sizeof(metadata_header));
  hdr.write(reinterpret_cast<const char*>(&tensor), sizeof(tensor));
  for(int64_t v: c_shape) hdr.write(reinterpret_cast<const char*>(&v), sizeof(int64_t));
  for(int64_t v: c_strides) hdr.write(reinterpret_cast<const char*>(&v), sizeof(int64_t));

  /* create return tuple ('bytes' hdr, sizeofdata) */
  PyObject * tuple = PyTuple_New(2);
  PyObject * bytes = PyByteArray_FromStringAndSize(hdr.str().c_str(), hdr.str().size());
  assert(bytes);

  if(PyTuple_SetItem(tuple, 0, bytes) != 0)
    throw General_exception("unexpected condition");

  if(PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong(data_size)) != 0)
    throw General_exception("unexpected condition");
  
  return tuple;
}


PyObject * pymmcore_dlpack_fix_pointers(PyObject * self,
                                        PyObject * args,
                                        PyObject * kwargs)
{
  static const char *kwlist[] = {"metadata",
                                 "value",
                                 NULL};

  PyObject * metadata_obj = nullptr;
  PyObject * value_obj = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "OO",
                                    const_cast<char**>(kwlist),
                                    &metadata_obj,
                                    &value_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyMemoryView_Check(metadata_obj) ||
      ! PyMemoryView_Check(value_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad argument types");
    return NULL;
  }
  
  Py_buffer * metadata_buffer = PyMemoryView_GET_BUFFER(metadata_obj);
  Py_buffer * value_buffer = PyMemoryView_GET_BUFFER(value_obj);

  if (!metadata_buffer || !value_buffer) {
    PyErr_SetString(PyExc_RuntimeError,"couldn't get memoryview buffers");
    return NULL;
  }

  if(! PyBuffer_IsContiguous(metadata_buffer,'C') ||
     ! PyBuffer_IsContiguous(value_buffer,'C')) {
    PyErr_SetString(PyExc_RuntimeError,"unexpected memoryview buffers");
    return NULL;
  }

  auto hdr = reinterpret_cast<MetaHeader*>(metadata_buffer->buf);
  auto dltensor = reinterpret_cast<DLTensor*>(&hdr[1]);
  dltensor->data = value_buffer->buf;

  auto p = reinterpret_cast<int64_t*>(&dltensor[1]);
  dltensor->shape = p;
  if(dltensor->strides == reinterpret_cast<int64_t*>(-1UL)) {
    dltensor->strides = p + dltensor->ndim;
  }
  
  PLOG("meta=%p value=%p", metadata_buffer->buf, value_buffer->buf);
  PLOG("dltensor->data=%p", dltensor->data);
  PLOG("dltensor->shape=%p", dltensor->shape);
  PLOG("dltensor->strides=%p", dltensor->strides);
  
  Py_RETURN_NONE;
}


PyObject * pymmcore_dlpack_as_str(PyObject * self,
                                  PyObject * args,
                                  PyObject * kwargs)
{
  static const char *kwlist[] = {"metadata",
                                 NULL};

  PyObject * metadata_obj = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &metadata_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyMemoryView_Check(metadata_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad argument types");
    return NULL;
  }
  
  Py_buffer * metadata_buffer = PyMemoryView_GET_BUFFER(metadata_obj);

  if (!metadata_buffer) {
    PyErr_SetString(PyExc_RuntimeError,"couldn't get memoryview buffers");
    return NULL;
  }

  if(! PyBuffer_IsContiguous(metadata_buffer,'C')) {
    PyErr_SetString(PyExc_RuntimeError,"unexpected memoryview buffers");
    return NULL;
  }

  auto hdr = reinterpret_cast<MetaHeader*>(metadata_buffer->buf);
  auto dltensor = reinterpret_cast<DLTensor*>(&hdr[1]);

  std::stringstream ss;
  ss << "<dlpack:DLTensor data=" << dltensor->data << " ndim=" << dltensor->ndim
     << " code=" << boost::numeric_cast<unsigned int>(dltensor->dtype.code)
     << " bits=" << boost::numeric_cast<unsigned int>(dltensor->dtype.bits)
     << " >";
  return PyUnicode_FromStringAndSize(ss.str().c_str(), ss.str().size());
}

static const char * kName = "dltensor";

extern "C" void PyCapsule_Destructor_function(PyObject * capsule)
{
  auto hdr = reinterpret_cast<MetaHeader*>(PyCapsule_GetContext(capsule));

  PLOG("dltensor refcnt=%u", hdr->refcnt);
  hdr->refcnt --; /* decrmeent reference count */
  PNOTICE("PyCapsule_Destructor_function !~!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
}

PyObject * pymmcore_dlpack_get_capsule(PyObject * self,
                                       PyObject * args,
                                       PyObject * kwargs)
{
  static const char *kwlist[] = {"metadata",
                                 NULL};

  PyObject * metadata_obj = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwargs,
                                    "O",
                                    const_cast<char**>(kwlist),
                                    &metadata_obj)) {
    PyErr_SetString(PyExc_RuntimeError,"bad arguments");
    return NULL;
  }

  if (! PyMemoryView_Check(metadata_obj)) {
      PyErr_SetString(PyExc_RuntimeError,"bad argument types");
    return NULL;
  }
  
  Py_buffer * metadata_buffer = PyMemoryView_GET_BUFFER(metadata_obj);

  if (!metadata_buffer) {
    PyErr_SetString(PyExc_RuntimeError,"couldn't get memoryview buffers");
    return NULL;
  }

  if(! PyBuffer_IsContiguous(metadata_buffer,'C')) {
    PyErr_SetString(PyExc_RuntimeError,"unexpected memoryview buffers");
    return NULL;
  }

  auto hdr = reinterpret_cast<MetaHeader*>(metadata_buffer->buf);

  hdr->refcnt += 1; /* does not need flusing */

  auto capsule = PyCapsule_New(reinterpret_cast<DLTensor*>(&hdr[1]), kName, &PyCapsule_Destructor_function);
  PyCapsule_SetContext(capsule, reinterpret_cast<void*>(hdr));
  return capsule;
}
