#ifndef __COMMON_TYPE_NAME_H__
#define __COMMON_TYPE_NAME_H__

#include <cxxabi.h>
#include <typeinfo>
#include <string>
#include <cstdlib>

#define type_of(X) type_name<decltype(X)>(X)

inline std::string demangle(const char* name) {

  int status = -4; // some arbitrary value to eliminate the compiler warning

  std::unique_ptr<char, void(*)(void*)> res {
    abi::__cxa_demangle(name, NULL, NULL, &status),
      std::free
      };

  return (status==0) ? res.get() : name ;
}

template <class T>
std::string type_name(const T& t) {
  
  return demangle(typeid(t).name());
}

#endif // __COMMON_TYPE_NAME_H__
