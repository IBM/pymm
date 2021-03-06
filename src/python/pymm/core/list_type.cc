#ifndef __PYMM_LIST_TYPE_H__
#define __PYMM_LIST_TYPE_H__

#include <limits.h>
#include <common/errors.h>
#include <common/logging.h>
#include <common/utils.h>

#include <ccpm/cca.h>
#include <ccpm/value_tracked.h>
#include <ccpm/container_cc.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Weffc++"
#include <EASTL/iterator.h>
#include <EASTL/list.h>
#pragma GCC diagnostic pop

#include <libpmem.h>
#include <Python.h>

typedef enum
  {
   SHELF_REFERENCE = 1,
   INLINE_FLOAT = 2,
   INLINE_LONGLONGINT = 3,
  } Element_type;

class Element
{
public:
  Element() {}
  Element(const Element_type& type_, const double value_) : type(type_), inline_float64(value_) {}
  Element(const Element_type& type_, const unsigned long tag_) : type(type_), tag(tag_) {}
  Element(const Element_type& type_, const long long int value_) : type(type_), inline_longlongint(value_) {}
  
  Element_type type;
  union {
    unsigned long tag;
    double        inline_float64;
    long long int inline_longlongint;
  };
};

using logged_ptr = ccpm::value_tracked<Element, ccpm::tracker_log>;
using cc_list_element = ccpm::container_cc<eastl::list<logged_ptr, ccpm::allocator_tl>>;

typedef struct {
  PyObject_HEAD
  ccpm::cca * heap;
  cc_list_element * list;
} List;

namespace
{
struct pmem_persister final
  : public ccpm::persister
{
  void persist(common::byte_span s) override
  {
    ::pmem_persist(::base(s), ::size(s));
  }
};
}

static pmem_persister persister;

static PyObject *
ListType_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
  auto self = (List *)type->tp_alloc(type, 0);
  assert(self);
  return (PyObject*)self;
}
/** 
 * tp_dealloc: Called when reference count is 0
 * 
 * @param self 
 */
static void
ListType_dealloc(List *self)
{
  delete self->heap;
  
  assert(self);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

/* convert the internally saved form into a Python object or reference */
static PyObject * present_element(const Element& element)
{
  PyObject* tuple = PyTuple_New(2);
  
  switch(element.type) {
  case SHELF_REFERENCE:
    PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong(element.tag));
    PyTuple_SetItem(tuple, 1, Py_True);
    break;
  case INLINE_FLOAT:
    PyTuple_SetItem(tuple, 0, PyFloat_FromDouble(element.inline_float64));
    PyTuple_SetItem(tuple, 1, Py_False);
    break;
  case INLINE_LONGLONGINT:
    PyTuple_SetItem(tuple, 0, PyLong_FromLongLong(element.inline_longlongint));
    PyTuple_SetItem(tuple, 1, Py_False);
    break;                    
  }

  return tuple;
}



static PyObject * ListType_method_getitem(List *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"item",
                                 NULL};

  PyObject * slice_criteria = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "O|",
                                    const_cast<char**>(kwlist),
                                    &slice_criteria)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType_method_getitem unable to parse args");
    return NULL;
  }

  if (PyLong_Check(slice_criteria)) {
    long list_size = self->list->container->size();
    long index = PyLong_AsLong(slice_criteria);
    if(index < 0L)
      index = list_size - labs(index);

    if(index < 0 or index >= list_size) {
      PyErr_SetString(PyExc_RuntimeError, "out of bounds item index");
      return NULL;
    }

    auto it = self->list->container->begin();
    while(index > 0) {
      it ++;
      index --;
    }
    assert(it != self->list->container->end());

    return present_element(*it);
  }
  
  Py_RETURN_NONE;
}


static PyObject * ListType_method_setitem(List *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"item",
                                 "value",
                                 "tag",
                                 NULL};

  long index;
  PyObject * value_obj = nullptr;
  unsigned long element_tag = 0;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "lO|k",
                                    const_cast<char**>(kwlist),
                                    &index,
                                    &value_obj,
                                    &element_tag)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType_method_setitem unable to parse args");
    return NULL;
  }

  long list_size = self->list->container->size();
  if(index < 0L)
    index = list_size - labs(index);

  if(index < 0L or index >= list_size) {
    PyErr_SetString(PyExc_RuntimeError, "out of bounds item index");
    return NULL;
  }

  /* locate element of list to modify */
  auto it = self->list->container->begin();
  while(index > 0) {
    it ++;
    index --;
  }
  assert(it != self->list->container->end());

  unsigned long removed_tag = it->tag;
  if(it->type != SHELF_REFERENCE) removed_tag = 0;

  if(element_tag > 0) { /* tag reference to an already shelved item */
    it->type = SHELF_REFERENCE;
    it->tag = element_tag;
    PLOG("Appended SHELF_REFERENCE (%lu)",element_tag);
    self->list->commit();
  }
  else {
    /* handle inlined types */
    if(PyLong_Check(value_obj)) { /* 64bit long number inline */
      int overflow = 0;
      long long value = PyLong_AsLongLongAndOverflow(value_obj, &overflow);
      if(overflow) {
        PyErr_SetString(PyExc_RuntimeError, "ListType_method_setitem long overflowed system's long long type");
        return NULL;
      }
      it->type = INLINE_LONGLONGINT;
      it->inline_longlongint = value;
    }
    else if(PyFloat_Check(value_obj)) { /* 64bit float inline */
      double value = PyFloat_AsDouble(value_obj);
      it->type = INLINE_FLOAT;
      it->inline_float64 = value;
    }
    else {
      PyErr_SetString(PyExc_RuntimeError, "ListType_method_setitem don't know how to append this type");
      return NULL;
    }
  }

  self->list->commit();
  return PyLong_FromUnsignedLong(removed_tag);
}


static PyObject * ListType_method_delitem(List *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"item",
                                 NULL};

  long index;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "l|",
                                    const_cast<char**>(kwlist),
                                    &index)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType_method_getitem unable to parse args");
    return NULL;
  }

  long list_size = self->list->container->size();
  if(index < 0L)
    index = list_size - labs(index);

  if(index < 0 or index >= list_size) {
    PyErr_SetString(PyExc_RuntimeError, "out of bounds item index");
    return NULL;
  }

  auto it = self->list->container->begin();
  while(index > 0) {
    it ++;
    index --;
  }
  assert(it != self->list->container->end());

  unsigned long removed_tag = it->tag;
  if(it->type != SHELF_REFERENCE) removed_tag = 0;

  self->list->container->erase(it);

  self->list->commit();
  /* return optional tag for removal from shelf */
  return PyLong_FromUnsignedLong(removed_tag);
}


static PyObject * ListType_method_append(List *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"element",
                                 "tag",
                                 NULL};

  unsigned long element_tag = 0;
  PyObject * element_object = nullptr;

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "O|k",
                                    const_cast<char**>(kwlist),
                                    &element_object,
                                    &element_tag)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType_method_append unable to parse args");
    return NULL;
  }

  assert(self->list);

  
  if(element_tag > 0) { /* add a reference to an already shelved item */
    self->list->container->push_back(Element(SHELF_REFERENCE,element_tag));
    PLOG("Appended SHELF_REFERENCE (%lu)",element_tag);
  }
  else {
    /* small types we tend to inline rather than using a reference to an
       instance in the pool */
    if(PyLong_Check(element_object)) {
      int overflow = 0;
      long long value = PyLong_AsLongLongAndOverflow(element_object, &overflow);
      if(overflow) {
        PyErr_SetString(PyExc_RuntimeError, "ListType_method_append long overflowed system's long long type");
        return NULL;
      }
      self->list->container->push_back(Element(INLINE_LONGLONGINT,value));
    }
    else if(PyFloat_Check(element_object)) {
      double value = PyFloat_AsDouble(element_object);
      self->list->container->push_back(Element(INLINE_FLOAT,value));
    }
    // else if(PyUnicode_Check(element_object)) {
    // }
    else {
      PyErr_SetString(PyExc_RuntimeError, "ListType_method_append don't know how to append this type");
      return NULL;
    }
  }
  self->list->commit();

  Py_RETURN_NONE;
}

PyObject * ListType_method_size(List *self, PyObject *args)
{
  return PyLong_FromUnsignedLong(self->list->container->size());
}



static PyMethodDef ListType_methods[] = 
  {
   {"append", (PyCFunction) ListType_method_append, METH_VARARGS | METH_KEYWORDS, "append(a) -> append 'a' to list"},
   {"getitem", (PyCFunction) ListType_method_getitem, METH_VARARGS | METH_KEYWORDS, "getitem(item) -> get element in list"},
   {"setitem", (PyCFunction) ListType_method_setitem, METH_VARARGS | METH_KEYWORDS, "setitem(item,value) -> set element in list"},
   {"delitem", (PyCFunction) ListType_method_delitem, METH_VARARGS | METH_KEYWORDS, "delitem(item) -> delete element from list"},
   {"size", (PyCFunction) ListType_method_size, METH_NOARGS, "size() -> get size of list"},
   {NULL}  /* Sentinel */
  };

static int
ListType_init(List *self, PyObject *args, PyObject *kwds)
{
  static const char *kwlist[] = {"buffer",
                                 "rehydrate",
                                 NULL};

  PyObject * memoryview_object = nullptr;
  int rehydrate = 0; /* zero for new construction */

  if (! PyArg_ParseTupleAndKeywords(args,
                                    kwds,
                                    "Op",
                                    const_cast<char**>(kwlist),
                                    &memoryview_object,
                                    &rehydrate)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType ctor unable to parse args");
    return -1;
  }

  if (! PyMemoryView_Check(memoryview_object)) {
    PyErr_SetString(PyExc_RuntimeError, "ListType ctor parameter is not a memory view");
    return -1;
  }
  Py_buffer * buffer = PyMemoryView_GET_BUFFER(memoryview_object);
  assert(buffer);

  /* create or rehydrate the heap */
  ccpm::region_vector_t rv(ccpm::region_vector_t::value_type(common::make_byte_span(buffer->buf,buffer->len)));
  if(rehydrate) {
    self->heap = new ccpm::cca(&persister, rv, ccpm::accept_all);
    void * root_base = ::base(self->heap->get_root());
    self->list = new (root_base) cc_list_element(std::move(*static_cast<cc_list_element*>(root_base)));
    self->list->rollback(); /* recovery check */
  }
  else {
    self->heap = new ccpm::cca(&persister, rv);
    void *root = self->heap->allocate_root(sizeof(cc_list_element)); //heap.allocate(list_size);
    assert(root);
    assert(self->heap);

    self->list = new (root) cc_list_element(&persister, *(self->heap));
  }

  return 0;
}


PyTypeObject ListType = {
                         PyVarObject_HEAD_INIT(NULL, 0)
                         "pymm.pymmcore.List",           /* tp_name */
                         sizeof(List)   ,      /* tp_basicsize */
                         0,                       /* tp_itemsize */
                         (destructor) ListType_dealloc,      /* tp_dealloc */
                         0,                       /* tp_print */
                         0,                       /* tp_getattr */
                         0,                       /* tp_setattr */
                         0,                       /* tp_reserved */
                         0,                       /* tp_repr */
                         0,                       /* tp_as_number */
                         0,                       /* tp_as_sequence */
                         0,                       /* tp_as_mapping */
                         0,                       /* tp_hash */
                         0,                       /* tp_call */
                         0,                       /* tp_str */
                         0,                       /* tp_getattro */
                         0,                       /* tp_setattro */
                         0,                       /* tp_as_buffer */
                         Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
                         "ListType",              /* tp_doc */
                         0,                       /* tp_traverse */
                         0,                       /* tp_clear */
                         0,                       /* tp_richcompare */
                         0,                       /* tp_weaklistoffset */
                         0,                       /* tp_iter */
                         0,                       /* tp_iternext */
                         ListType_methods,        /* tp_methods */
                         0, //ListType_members,         /* tp_members */
                         0,                       /* tp_getset */
                         0,                       /* tp_base */
                         0,                       /* tp_dict */
                         0,                       /* tp_descr_get */
                         0,                       /* tp_descr_set */
                         0,                       /* tp_dictoffset */
                         (initproc)ListType_init,  /* tp_init */
                         0,            /* tp_alloc */
                         ListType_new,             /* tp_new */
                         0, /* tp_free */
};



#endif
