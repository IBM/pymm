# Drop to 14 if necessary avoid boost bug (surfaced by g++ 7.1 std=c++1z) icl/type_traits/type_to_string.hpp partial specialization
find_package(Boost 1.66.0)
if(NOT Boost_FOUND)
set(CMAKE_CXX_STANDARD 14)
endif()
