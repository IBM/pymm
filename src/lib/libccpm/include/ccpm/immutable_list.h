#ifndef __CCPM_IMMUTABLE_LIST__
#define __CCPM_IMMUTABLE_LIST__

#include <common/type_name.h>
#include <ccpm/interfaces.h>
#include <ccpm/immutable_allocator.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include <EASTL/list.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <nop/serializer.h>
#include <nop/utility/stream_reader.h>
#pragma GCC diagnostic pop

namespace ccpm
{
template <class T>
class Immutable_list
{
public:
  using list_type = eastl::list<T, EASTL_immutable_allocator>;
  using allocator_type = ccpm::EASTL_immutable_allocator;
  using size_type = typename list_type::size_type;
  using iterator = typename list_type::iterator;

  explicit Immutable_list(const ccpm::region_span regions, bool force_init = false)
    : _allocator_base(regions, nullptr, force_init),
      _allocator(&_allocator_base)
  {
    if(force_init || _allocator_base.rebuilt()) {
      _list = new (_allocator.allocate(sizeof(list_type))) list_type(_allocator);
    }
    else {
      _list = reinterpret_cast<list_type*>(_allocator_base.first_element());
      _list->set_allocator(_allocator);
    }
        
    assert(_list);
    _allocator_base.persist();
  }

  /* get string version of this type instance */
  static std::string type_name() { return
      std::string("ccpm::Immutable_list<") +
      demangle(typeid(T).name()) + ">"; }
  
  /* const read only methods */
  inline size_type size() const noexcept { return _list->size(); }

  /* non-const methods ; TODO add crash consistency */
  inline void push_front( const T& value ) { _list->push_front(value);  }
  inline void push_back( const T& value ) { _list->push_back(value); }
  inline void sort() { _list->sort(); }
  inline iterator begin() noexcept { return _list->begin(); }
  inline iterator end() noexcept { return _list->end(); }

  /* TODO add passthru methods */
 private:
  Immutable_allocator_base  _allocator_base;
  allocator_type            _allocator;
  list_type *               _list; /* don't use smart pointers */
};

template <class T>
class Immutable_list_dispatcher : public Immutable_list<T>
{
public:
  explicit Immutable_list_dispatcher(const ccpm::region_vector_t& regions, bool force_init = false)
    : Immutable_list<T>(regions, force_init) {
  }
  
  /* dynamic invocation methods */
  void push_front(const std::string& value) {
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{value};
    T typed_value;
    deserializer.Read(&typed_value);
    push_front(typed_value);
  }

  void push_back(const std::string& value) {
    nop::Deserializer<nop::StreamReader<std::stringstream>> deserializer{value};
    T typed_value;
    deserializer.Read(&typed_value);
    push_back(typed_value);
  }  
};

  
} // namespace

#endif
