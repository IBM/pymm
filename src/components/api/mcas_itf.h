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

#ifndef __API_MCAS_ITF__
#define __API_MCAS_ITF__

#include <api/components.h>
#include <api/kvindex_itf.h>
#include <api/kvstore_itf.h>
#include <boost/optional.hpp>
#include <common/byte_span.h>
#include <common/pointer_cast.h>
#include <common/string_view.h>
#include <gsl/span>

#include <array>
#include <cstdint> /* uint16_t */
#include <memory>

#define DECLARE_OPAQUE_TYPE(NAME)               \
  struct Opaque_##NAME {                        \
    virtual ~Opaque_##NAME() {}                 \
  }

namespace component
{
/**
 * mcas client interface (this will include both KV and AS capabilities)
 */

class IMCAS : public component::IBase,
              public KVStore
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0x33af1b99,0xbc51,0x49ff,0xa27b,0xd4,0xe8,0x19,0x03,0xbb,0x02);
  // clang-format on

public:
  DECLARE_OPAQUE_TYPE(async_handle);

  using async_handle_t  = Opaque_async_handle*;
  using pool_t          = component::KVStore::pool_t;
  using key_t           = KVStore::key_t;
  using Attribute       = KVStore::Attribute;
  using Addr            = KVStore::Addr;

  static constexpr async_handle_t  ASYNC_HANDLE_INIT  = nullptr;

  enum {
        FLAGS_NONE        = KVStore::FLAGS_NONE,
        FLAGS_READ_ONLY   = KVStore::FLAGS_READ_ONLY,
        FLAGS_SET_SIZE    = KVStore::FLAGS_SET_SIZE,
        FLAGS_CREATE_ONLY = KVStore::FLAGS_CREATE_ONLY,
        FLAGS_DONT_STOMP  = KVStore::FLAGS_DONT_STOMP,
        FLAGS_NO_RESIZE   = KVStore::FLAGS_NO_RESIZE,
        FLAGS_MAX_VALUE   = KVStore::FLAGS_MAX_VALUE,
  };


  /* per-shard statistics */
  struct Shard_stats {
    uint64_t op_request_count;
    uint64_t op_put_count;
    uint64_t op_get_count;
    uint64_t op_put_direct_count;
    uint64_t op_get_direct_count;
    uint64_t op_get_twostage_count;
    uint64_t op_ado_count;
    uint64_t op_erase_count;
    uint64_t op_get_direct_offset_count;
    uint64_t op_failed_request_count;
    uint64_t last_op_count_snapshot;
    uint16_t client_count;

  public:
    Shard_stats()
      : op_request_count(0)
      , op_put_count(0), op_get_count(0), op_put_direct_count(0), op_get_direct_count(0), op_get_twostage_count(0)
      , op_ado_count(0), op_erase_count(0), op_get_direct_offset_count(0)
      , op_failed_request_count(0), last_op_count_snapshot(0), client_count(0)
    {
    }
  } __attribute__((packed));

  using ado_flags_t = uint32_t;

  static constexpr ado_flags_t ADO_FLAG_NONE = 0;
  /*< operation is asynchronous */
  static constexpr ado_flags_t ADO_FLAG_ASYNC = (1 << 0);
  /*< create KV pair if needed */
  static constexpr ado_flags_t ADO_FLAG_CREATE_ON_DEMAND = (1 << 1);
  /*< create only - allocate key,value but don't call ADO */
  static constexpr ado_flags_t ADO_FLAG_CREATE_ONLY = (1 << 2);
  /*< do not overwrite value if it already exists */
  static constexpr ado_flags_t ADO_FLAG_NO_OVERWRITE = (1 << 3);
  /*< create value but do not attach to key, unless key does not exist */
  static constexpr ado_flags_t ADO_FLAG_DETACHED = (1 << 4);
  /*< only take read lock */
  static constexpr ado_flags_t ADO_FLAG_READ_ONLY = (1 << 5);
  /*< zero any newly allocated value memory */
  static constexpr ado_flags_t ADO_FLAG_ZERO_NEW_VALUE = (1 << 6);
  /*< internal use only: on return provide IO response */
  static constexpr ado_flags_t ADO_FLAG_INTERNAL_IO_RESPONSE = (1 << 7);
  /*< internal use only: on return provide IO response with value buffer */
  static constexpr ado_flags_t ADO_FLAG_INTERNAL_IO_RESPONSE_VALUE = (1 << 8);

public:
  /**
   * Determine thread safety of the component
   *
   *
   * @return THREAD_MODEL_XXXX
   */
  virtual int thread_safety() const = 0;

  /**
   * If the ADO is configured for the shard then the ADO process is
   * instantiated "attached" to the pool.
   *
   * The "base" parameter is unused.
   */
  using KVStore::create_pool;
  using KVStore::open_pool;

  using KVStore::delete_pool;

  /**
   * Close and delete an existing pool from a pool handle. Only one
   * reference count should exist. Any ADO plugin is notified before
   * the pool is deleted.
   *
   * @param pool Pool handle
   *
   * @return S_OK or E_BUSY if reference count > 1
   */
  virtual status_t delete_pool(const pool_t pool) = 0;

  /**
   * Configure a pool
   *
   * @param setting Configuration request (e.g., AddIndex::VolatileTree)

   *
   * @return S_OK on success
   */
  virtual status_t configure_pool(const pool_t pool, const std::string& setting) = 0;

  using KVStore::put;

  virtual status_t put(const pool_t pool,
                       const std::string&  key,
                       const std::string&  value,
                       const unsigned int  flags = FLAGS_NONE)
  {
    /* this does not store any null terminator */
    return put(pool, key, value.data(), value.length(), flags);
  }

  /*
   * Asynchronous put operation.  Use check_async_completion to check for
   * completion. This operation is not normally used, simple put is fast.
   *
   * @param pool Pool handle
   * @param key Object key
   * @param value Value

   * @param value_len Value length in bytes
   * @param out_handle Async work handle
   * @param flags Optional flags
   *
   * @return S_OK or other error code
   */
  virtual status_t async_put(const IMCAS::pool_t pool,
                             const std::string&  key,
                             const void*         value,
                             const size_t        value_len,
                             async_handle_t&     out_handle,
                             const unsigned int  flags = IMCAS::FLAGS_NONE) = 0;

  virtual status_t async_put(const IMCAS::pool_t pool,
                             const std::string&  key,
                             const std::string&  value,
                             async_handle_t&     out_handle,
                             const unsigned int  flags = IMCAS::FLAGS_NONE)
  {
    (void)out_handle; // unused
    return async_put(pool, key, value.data(), value.length(), out_handle, flags);
  }

  /**
   * Zero-copy only if value size > ~2MiB or FORCE_DIRECT=1 is set.
   */
  using KVStore::put_direct;

  /**
   * Asynchronous put_direct operation.  Use check_async_completion to check for
   * completion.
   *
   * @param pool Pool handle
   * @param key Object key
   * @param value Value
   * @param value_len Value length in bytes
   * @param handle Memory registration handle
   * @param out_handle Async handle
   * @param flags Optional flags
   *
   * @return S_OK or other error code
   */
  virtual status_t async_put_direct(const IMCAS::pool_t   pool,
                                    const std::string&    key,
                                    const void*           value,
                                    const size_t          value_len,
                                    async_handle_t&       out_handle,
                                    const memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE,
                                    const unsigned int    flags  = IMCAS::FLAGS_NONE)
  {
    return
      async_put_direct(
                       pool
                       , key
                       , std::array<common::const_byte_span,1>{common::make_const_byte_span(value, value_len)}
                       , out_handle, std::array<memory_handle_t,1>{handle}
                       , flags
                       );
  }

  /**
   * Asynchronous put_direct operation.  Use check_async_completion to check for
   * completion.
   *
   * @param pool Pool handle
   * @param key Object key
   * @param values list of value sources
   * @param out_handle Async handle
   * @param handle List of memory registration handle
   * @param flags Optional flags
   *
   * @return S_OK or other error code
   */
  virtual status_t async_put_direct(const IMCAS::pool_t   pool,
                                    const std::string&    key,
                                    gsl::span<const common::const_byte_span> value,
                                    async_handle_t&       out_handle,
                                    gsl::span<const memory_handle_t> handles = gsl::span<const memory_handle_t>(),
                                    const unsigned int    flags  = IMCAS::FLAGS_NONE) = 0;

  using KVStore::get;

  virtual status_t get(const pool_t pool,
                       const std::string& key,
                       std::string& out_value)
  {
    void*  val      = nullptr;
    size_t val_size = 0;
    auto   s        = this->get(pool, key, val, val_size);

    /* copy result */
    if (s == S_OK) {
      out_value.assign(static_cast<char*>(val), val_size);
      this->free_memory(val);
    }
    return s;
  }

  /**
   * Asynchronously read an object value directly into client-provided memory.
   *
   * @param pool Pool handle
   * @param key Object key
   * @param out_value Client provided buffer for value
   * @param out_value_len [in] size of value memory in bytes [out] size of value
   * @param out_handle Async work handle
   * @param handle Memory registration handle
   *
   * @return S_OK, S_MORE if only a portion of value is read, E_BAD_ALIGNMENT on
   * invalid alignment, or other error code
   */
  virtual status_t async_get_direct(const IMCAS::pool_t          pool,
                                    const std::string&           key,
                                    void*                        out_value,
                                    size_t&                      out_value_len,
                                    async_handle_t&              out_handle,
                                    const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) = 0;

  /**
   * Read memory directly into client-provided memory.
   *
   * @param pool Pool handle
   * @param offset offset within ithe concatenation of the pool's memory regions
   * @param size requested size (becomes available size)
   * @param out_buffer Client provided buffer for value
   * @param out_handle Async work handle
   * @param handle Memory registration handle
   *
   * @return S_OK, or error code
   */
  virtual status_t async_get_direct_offset(const IMCAS::pool_t pool,
                                           const offset_t offset,
                                           size_t &size,
                                           void* out_buffer,
                                           async_handle_t& out_handle,
                                           const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) = 0;

  virtual status_t get_direct_offset(const IMCAS::pool_t pool,
                                     const offset_t offset,
                                     size_t &size,
                                     void* out_buffer,
                                     const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) = 0;

  /**
   * Write memory directly into client-provided memory.
   *
   * @param pool Pool handle
   * @param offset offset within ithe concatenation of the pool's memory regions
   * @param size offered size (becomes available size)
   * @param buffer Client provided value
   * @param out_handle Async work handle
   * @param handle Memory registration handle
   *
   * @return S_OK, or error code
   */
  virtual status_t async_put_direct_offset(const IMCAS::pool_t pool,
                                           const offset_t offset,
                                           size_t &size,
                                           const void *const buffer,
                                           async_handle_t& out_handle,
                                           const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) = 0;

  virtual status_t put_direct_offset(const IMCAS::pool_t pool,
                                     const offset_t offset,
                                     size_t &size,
                                     const void *const buffer,
                                     const IMCAS::memory_handle_t handle = IMCAS::MEMORY_HANDLE_NONE) = 0;

  /**
   * Check for completion from asynchronous invocation
   *
   * @param handle Asynchronous work handle.
   *
   * @return S_OK or E_BUSY if not yet complete
   */
  virtual status_t check_async_completion(async_handle_t& handle) = 0;

  /**
   * Perform key search based on regex or prefix
   *
   * @param pool Pool handle
   * @param key_expression Regular expression or prefix (e.g. "prefix:carKey")
   * @param offset Offset from which to search
   * @param out_matched_offset Out offset of match
   * @param out_keys Out vector of matching keys
   *
   * @return S_OK on success
   */
  virtual status_t find(const IMCAS::pool_t pool,
                        const std::string&  key_expression,
                        const offset_t      offset,
                        offset_t&           out_matched_offset,
                        std::string&        out_matched_key) = 0;

  /**
   * Erase an object asynchronously
   *
   * @param pool Pool handle
   * @param key Object key
   * @param out_handle Async work handle
   *
   * @return S_OK or error code
   */
  virtual status_t async_erase(const IMCAS::pool_t pool, const std::string& key, async_handle_t& out_handle) = 0;

  /**
   * Retrieve shard statistics
   *
   * @param out_stats
   *
   * @return S_OK on success
   */
  virtual status_t get_statistics(Shard_stats& out_stats) = 0;

  /**
   * ADO_response data structure manages response data sent back from the ADO
   * invocations.  The free function is so we can eventually support zero-copy.
   * The layer id identifies which ADO plugin the response came from.
   */
  class ADO_response {

  private:

#pragma GCC diagnostic push /* pointer members are considered inefficient */
#pragma GCC diagnostic ignored "-Weffc++"

    class Data_reference {
    public:
      Data_reference(void * data) : _data(data) { assert(data); }
      Data_reference() = delete;
      virtual ~Data_reference() { assert(_data); ::free(_data); }
      void * _data;
    };

#pragma GCC diagnostic pop

  public:
    ADO_response() = delete;

    ADO_response(void* data, size_t data_len, uint32_t layer_id)
      : _ref(std::make_shared<Data_reference>(data)),
        _data_len(data_len),
        _layer_id(layer_id) {}

    ADO_response(ADO_response&& src) noexcept
      : _ref(src.datasp()),
        _data_len(src.data_len()),
        _layer_id(src.layer_id()) {}

    inline std::string str() const { return std::string(data(), data_len()); }
    inline const char* data() const { return static_cast<const char*>(_ref->_data); }
    inline size_t data_len() const { return _data_len; }
    inline uint32_t layer_id() const { return _layer_id; }
    inline std::shared_ptr<Data_reference>& datasp() { return _ref; }

    template <typename T>
    inline T* cast_data() const  {  return static_cast<T*>(_ref->_data);   }

  private:
    std::shared_ptr<Data_reference> _ref; /* smart pointer */
    size_t                          _data_len;
    uint32_t                        _layer_id; /* optional layer identifier */
  };


  /**
   * Used to invoke an operation on an active data object
   *
   * Roughly, the shard locates a data value by key and calls ADO (with accessors to both the key and data).
   *   Several variations:
   *       1) Skip the "locate data" operation (could have been directed by a null
   *          key address in the basic_string_view form, or by a flag, but is
   *          instead indicated by a zero-length key)
   *       2) Skip the ADO call (directed by ADO_FLAG_CREATE_ONLY)
   *       3) Create uninitialized data of size value_size if the key was not found
   *          (directed by ADO_FLAG_CREATE_ON_DEMAND)
   *       4) Create uninitialized data of size value_size if the key was not found(?)
   *          not associated with the key and maybe also create a "root value" which
   *          which is attached to the key (directed by ADO_FLAG_DETACHED)
   *
   * @param pool Pool handle
   * @param key Key. Note, if key is empty, the work request is key-less.
   * @param request Request data
   * @param request_len Length of request in bytes
   * @param flags Flags for invocation (see ADO_FLAG_CREATE_ONLY, ADO_FLAG_READ_ONLY)
   * @param out_response Responses from invocation
   * @param value_size Optional parameter to define value size to create for
   * on-demand
   *
   * @return S_OK on success
   */
  virtual status_t invoke_ado(const IMCAS::pool_t           pool,
                              const basic_string_view<byte> key,
                              const basic_string_view<byte> request,
                              const ado_flags_t             flags,
                              std::vector<ADO_response>&    out_response,
                              const size_t                  value_size = 0) = 0;

  virtual status_t invoke_ado(const IMCAS::pool_t        pool,
                              const std::string&         key,
                              const void*                request,
                              const size_t               request_len,
                              const ado_flags_t          flags,
                              std::vector<ADO_response>& out_response,
                              const size_t               value_size = 0)
  {
    return
      invoke_ado(pool,
                 basic_string_view<byte>(common::pointer_cast<const byte>(key.data()), key.size()),
                 basic_string_view<byte>(static_cast<const byte *>(request), request_len),
                 flags, out_response, value_size);
  }

  inline status_t invoke_ado(const IMCAS::pool_t        pool,
                             const std::string&         key,
                             const std::string&         request,
                             const ado_flags_t          flags,
                             std::vector<ADO_response>& out_response,
                             const size_t               value_size = 0)
  {
    return
      invoke_ado(pool, key, request.data(), request.length(), flags, out_response, value_size);
  }


  /**
   * Used to asynchronously invoke an operation on an ADO
   *
   * Roughly, the shard locates a data value by key and calls ADO (with accessors to both the key and data).
   *
   * @param pool Pool handle
   * @param key Key. Note, if key is empty, the work request is key-less.
   * @param request Request data
   * @param request_len Length of request in bytes
   * @param flags Flags for invocation (see ADO_FLAG_XXX)
   * @param out_response Response passed back from ADO invocation
   * @param out_async_handle Handle to task for later result collection
   * @param value_size Optional parameter to define value size to create for on-demand
   *
   * @return S_OK on success
   */
  virtual status_t async_invoke_ado(const IMCAS::pool_t           pool,
                                    const basic_string_view<byte> key,
                                    const basic_string_view<byte> request,
                                    const ado_flags_t             flags,
                                    std::vector<ADO_response>&    out_response,
                                    async_handle_t&               out_async_handle,
                                    const size_t                  value_size = 0) = 0;

  inline status_t async_invoke_ado(const IMCAS::pool_t        pool,
                                   const std::string&         key,
                                   const std::string&         request,
                                   const ado_flags_t          flags,
                                   std::vector<ADO_response>& out_response,
                                   async_handle_t&            out_async_handle,
                                   const size_t               value_size = 0)
  {
    return
      async_invoke_ado(pool,
                       basic_string_view<byte>(common::pointer_cast<const byte>(key.data()), key.size()),
                       basic_string_view<byte>(common::pointer_cast<const byte>(request.data()), request.length()),
                       flags, out_response, out_async_handle, value_size);
  }

  inline status_t async_invoke_ado(const IMCAS::pool_t        pool,
                                   const std::string&         key,
                                   const void*                request,
                                   const size_t               request_len,
                                   const ado_flags_t          flags,
                                   std::vector<ADO_response>& out_response,
                                   async_handle_t&            out_async_handle,
                                   const size_t               value_size = 0)
  {
    return
      async_invoke_ado(pool,
                       basic_string_view<byte>(common::pointer_cast<const byte>(key.data()), key.size()),
                       basic_string_view<byte>(static_cast<const byte *>(request), request_len),
                       flags, out_response, out_async_handle, value_size);
  }


  /**
   * Used to invoke a combined put + ADO operation on an active data object.
   *
   * Roughly, the shard writes a data value by key and calls ADO (with accessors to both the key and data).
   *
   * @param pool Pool handle
   * @param key Key
   * @param request Request data
   * @param request_len Length of request data in bytes
   * @param value Value data
   * @param value_len Length of value data in bytes
   * @param root_len Length to allocate for root value (with ADO_FLAG_DETACHED)
   * @param flags Flags for invocation (ADO_FLAG_NO_OVERWRITE, ADO_FLAG_DETACHED)
   * @param out_response Response passed back from ADO invocation
   *
   * @return S_OK on success
   */
  virtual status_t invoke_put_ado(const IMCAS::pool_t           pool,
                                  const basic_string_view<byte> key,
                                  const basic_string_view<byte> request,
                                  const basic_string_view<byte> value,
                                  const size_t                  root_len,
                                  const ado_flags_t             flags,
                                  std::vector<ADO_response>&    out_response) = 0;

  virtual status_t invoke_put_ado(const IMCAS::pool_t        pool,
                                  const std::string&         key,
                                  const void*                request,
                                  const size_t               request_len,
                                  const void*                value,
                                  const size_t               value_len,
                                  const size_t               root_len,
                                  const ado_flags_t          flags,
                                  std::vector<ADO_response>& out_response)
  {
    return
      invoke_put_ado(pool,
                     basic_string_view<byte>(common::pointer_cast<const byte>(key.data()), key.size()),
                     basic_string_view<byte>(static_cast<const byte *>(request), request_len),
                     basic_string_view<byte>(static_cast<const byte *>(value), value_len),
                     root_len, flags, out_response);
  }

  inline status_t invoke_put_ado(const IMCAS::pool_t        pool,
                                 const std::string&         key,
                                 const std::string&         request,
                                 const std::string&         value,
                                 const size_t               root_len,
                                 const ado_flags_t          flags,
                                 std::vector<ADO_response>& out_response)
  {
    return invoke_put_ado(pool,
                          key,
                          request.data(),
                          request.length(),
                          value.data(),
                          value.length(),
                          root_len,
                          flags,
                          out_response);
  }

  /**
   * Used to asynchronously invoke a combined put + ADO operation on an active data object.
   *
   * Roughly, the shard writes a data value by key and calls ADO (with accessors to both the key and data).
   *
   * @param pool Pool handle
   * @param key Key
   * @param request Request data
   * @param request_len Length of request data in bytes
   * @param value Value data
   * @param value_len Length of value data in bytes
   * @param root_len Length to allocate for root value (with ADO_FLAG_DETACHED)
   * @param flags Flags for invocation (ADO_FLAG_NO_OVERWRITE, ADO_FLAG_DETACHED)
   * @param out_response Responses from ADO invocation
   * @param out_async_handle Handle to task for later result collection
   *
   * @return S_OK on success
   */
  virtual status_t async_invoke_put_ado(const IMCAS::pool_t           pool,
                                        const basic_string_view<byte> key,
                                        const basic_string_view<byte> request,
                                        const basic_string_view<byte> value,
                                        const size_t                  root_len,
                                        const ado_flags_t             flags,
                                        std::vector<ADO_response>&    out_response,
                                        async_handle_t&               out_async_handle) = 0;

  virtual status_t async_invoke_put_ado(const IMCAS::pool_t        pool,
                                        const std::string&         key,
                                        const void*                request,
                                        const size_t               request_len,
                                        const void*                value,
                                        const size_t               value_len,
                                        const size_t               root_len,
                                        const ado_flags_t          flags,
                                        std::vector<ADO_response>& out_response,
                                        async_handle_t&            out_async_handle)
  {
    return
      async_invoke_put_ado(pool,
                           basic_string_view<byte>(common::pointer_cast<const byte>(key.data()), key.size()),
                           basic_string_view<byte>(static_cast<const byte *>(request), request_len),
                           basic_string_view<byte>(static_cast<const byte *>(value), value_len),
                           root_len, flags, out_response, out_async_handle);
  }

  inline status_t async_invoke_put_ado(const IMCAS::pool_t        pool,
                                       const std::string&         key,
                                       const std::string&         request,
                                       const std::string&         value,
                                       const size_t               root_len,
                                       const ado_flags_t          flags,
                                       std::vector<ADO_response>& out_response,
                                       async_handle_t&            out_async_handle)
  {
    return async_invoke_put_ado(pool,
                                key,
                                request.data(),
                                request.length(),
                                value.data(),
                                value.length(),
                                root_len,
                                flags,
                                out_response,
                                out_async_handle);
  }



  /**
   * Debug routine
   *
   * @param pool Pool handle
   * @param cmd Debug command
   * @param arg Parameter for debug operation
   */
  virtual void debug(const IMCAS::pool_t pool, const unsigned cmd, const uint64_t arg) = 0;
};


class IMCAS_factory : public IKVStore_factory {
  using string_view = common::string_view;
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacf1b99,0xbc51,0x49ff,0xa27b,0xd4,0xe8,0x19,0x03,0xbb,0x02);
  // clang-format on

  /**
   * Create a session to a remote shard
   *
   * @param debug_level         Debug level (0-3)
   * @param patience            Time out patience in seconds
   * @param owner               Owner info (not used)
   * @param src_nic_device      Client-side network device (e.g., mlx5_0, eth0)
   * @param src_ip_addr         Client-side IP address
   * @param dest_addr_with_port Server-side IP address and port (e.g. 10.0.0.21:11911, 9.1.75.6:11911:sockets)
   * @param other               Other optional parameters (e.g. { "security":"tls:auth" })
   *
   * @return Pointer to IMCAS instance. Use release_ref() to close.
   */
  virtual IMCAS* mcas_create_nsd(const unsigned, // debug_level
                                 const unsigned, // patience
                                 const string_view, // owner
                                 const string_view, // src_nic_device
                                 const string_view, // src_ip_addr
                                 const string_view, // dest_addr_with_port
                                 const string_view = string_view()) // other
  {
    throw API_exception("IMCAS_factory::mcas_create(debug_level,patience,owner,addr_with_port,nic_device) not implemented");
  }

  /** 
   * Create a session to a remote shard (alternative)
   * 
   * @param debug_level          Debug level
   * @param patience             Timeout patience in seconds
   * @param owner                Owner information (not used)
   * @param dest_addr_with_port  Destination server IP address and port
   * @param nic_device           Local NIC device to use (e.g., mlx5_0, eth0)
   * @param other                Other optional parameters (e.g. { "security":"tls:auth" })
   * 
   * @return Pointer to IMCAS instance. Use release_ref() to close.
   */
  IMCAS* mcas_create(const unsigned    debug_level,
                     const unsigned    patience,
                     const string_view owner,
                     const string_view dest_addr_with_port,
                     const string_view nic_device,
                     const string_view other = string_view())
  {
    return mcas_create_nsd(debug_level, patience, owner, nic_device, string_view(), dest_addr_with_port, other);
  }
};

}  // namespace component

#endif
