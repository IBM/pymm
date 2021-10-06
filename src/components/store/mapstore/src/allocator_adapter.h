#ifndef __ALLOCATOR_ADAPTER_H__
#define __ALLOCATOR_ADAPTER_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#include <stdexcept>
#include "mm_plugin_itf.h"

template <class T, class Alloc_T>
class allocator_adapter {
public:
  using value_type = T;

//     using pointer       = value_type*;
//     using const_pointer = typename std::pointer_traits<pointer>::template
//                                                     rebind<value_type const>;
//     using void_pointer       = typename std::pointer_traits<pointer>::template
//                                                           rebind<void>;
//     using const_void_pointer = typename std::pointer_traits<pointer>::template
//                                                           rebind<const void>;

//     using difference_type = typename std::pointer_traits<pointer>::difference_type;
//     using size_type       = std::make_unsigned_t<difference_type>;

//     template <class U> struct rebind {typedef allocator<U> other;};

  allocator_adapter() noexcept {}  // not required, unless used

  template <class T_, class Alloc_T_>
  allocator_adapter(allocator_adapter<T_, Alloc_T_> const&) noexcept {}

  value_type*  // Use pointer if pointer is not a value_type*
  allocate(std::size_t n)
  {
    return static_cast<value_type*>(::operator new (n*sizeof(value_type)));
  }

  void
  deallocate(value_type* p, std::size_t) noexcept  // Use pointer if pointer is not a value_type*
  {
    ::operator delete(p);
  }

  //     value_type*
//     allocate(std::size_t n, const_void_pointer)
//     {
//         return allocate(n);
//     }

//     template <class U, class ...Args>
//     void
//     construct(U* p, Args&& ...args)
//     {
//         ::new(p) U(std::forward<Args>(args)...);
//     }

//     template <class U>
//     void
//     destroy(U* p) noexcept
//     {
//         p->~U();
//     }

//     std::size_t
//     max_size() const noexcept
//     {
//         return std::numeric_limits<size_type>::max();
//     }

//     allocator
//     select_on_container_copy_construction() const
//     {
//         return *this;
//     }

//     using propagate_on_container_copy_assignment = std::false_type;
//     using propagate_on_container_move_assignment = std::false_type;
//     using propagate_on_container_swap            = std::false_type;
//     using is_always_equal                        = std::is_empty<allocator>;
};

template <class T, class U, class Alloc_T>
bool
operator==(allocator_adapter<T,Alloc_T> const&, allocator_adapter<U, Alloc_T> const&) noexcept
{
    return true;
}

template <class T, class U, class Alloc_T>
bool
operator!=(allocator_adapter<T, Alloc_T> const& x, allocator_adapter<U, Alloc_T> const& y) noexcept
{
    return !(x == y);
}

#pragma GCC diagnostic pop

#endif // __ALLOCATOR_ADAPTER__
