#ifndef __CCPM_BASIC_TYPES__
#define __CCPM_BASIC_TYPES__

namespace ccpm
{


template<typename T>
class Basic_integer_type
{
public:
  Basic_integer_type() {
    if(check_aligned(this, sizeof(T)) == false)
       throw std::invalid_argument("Basic_type expect alignment");
  }
  
  const T& operator=(const T& value) {
    _value = value;
    flush();
    return _value;
  }
  T& operator++() { _value += 1; flush(); return _value;  }
  T& operator+=(int x) { _value += x; flush(); return _value;  }
  T& operator--(int x) { _value -= x; flush(); return _value;  }
  operator T() const { return _value; }
  
private:
  inline void flush() { pmem_flush(&_value, sizeof(T)); }
  T _value;
};


using Uint64 = Basic_integer_type<uint64_t>;
using Int64 = Basic_integer_type<int64_t>;

}

#endif // __CCPM_BASIC_TYPES__
