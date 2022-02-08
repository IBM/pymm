/*
   Copyright [2020] [IBM Corporation]
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

#if ! defined _NUPM_FILESYSTEM_STD_ && defined __has_include
  #if __has_include (<filesystem>) && __cplusplus >= 201703L
    #include <filesystem>
    #define _NUPM_FILESYSTEM_STD_ 1
  #endif
#endif
#if ! defined _NUPM_FILESYSTEM_STD_
  #include <experimental/filesystem>
  #define _NUPM_FILESYSTEM_STD_ 0
#endif
