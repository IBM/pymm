/*
   Copyright [2017-2020] [IBM Corporation]
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

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#ifndef __API_CRYPTO_ITF__
#define __API_CRYPTO_ITF__

#include <string>
#include <map>
#include <functional>
#include <api/components.h>
#include <common/types.h>

namespace component
{
#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wunused-parameter"

/**
 * ICrypto security services for MCAS
 */
class ICrypto : public component::IBase {

public:
  typedef enum { /* matches gnutls for ease of translation */
    MAC_MD5 = 2,
    MAC_SHA1 = 3,
    MAC_RMD160 = 4,
    MAC_MD2 = 5,
    MAC_SHA256 = 6,
    MAC_SHA384 = 7,
    MAC_SHA512 = 8,
    MAC_SHA224 = 9,
    MAC_AEAD = 200 /* indicates that MAC is on the cipher */
  } mac_algorithm_t;

  typedef enum {
    CIPHER_AES_128_GCM,
    CIPHER_AES_256_GCM,
  } cipher_t;

public:
  DECLARE_INTERFACE_UUID(0xD7F80DD6,0xBB71,0x48C3,0xAF6E,0x2C,0x2E,0xFD,0x7A,0x3B,0xD7);

public:
  class Crypto_session {
  public:
    virtual status_t shutdown() = 0;
    virtual bool is_server_side() const = 0;
    virtual const std::string& client_uuid() const = 0;
    virtual ~Crypto_session() {}
  };

  using session_t  = Crypto_session*;

  /**
   * @brief      Initialize cipher suite, cert and key files
   *
   * @param[in]  cipher_suite  The cipher suite
   * @param[in]  cert_file     The cert file
   * @param[in]  key_file      The key file
   *
   * @return     S_OK on success
   */
  virtual status_t initialize(const std::string& cipher_suite,
                              const std::string& cert_file,
                              const std::string& key_file) = 0;


  /**
   * @brief      Server-side accept certificate-based session
   *
   * @param[in]  ip_addr Network interface IP addr
   * @param[in]  port  Network port
   * @param[in]  timeout  Timeout in milliseconds
   *
   * @return     Session handle (free with close_session)
   */
  virtual session_t accept_cert_session(const std::string& ip_addr,
                                        const int port,
                                        const unsigned int timeout_ms) = 0;

  /**
   * @brief      Client-side open certificate-based session
   *
   * @param[in]  cipher_suite  Cipher suite
   * @param[in]  server_ip     Server ip
   * @param[in]  server_port   Server port
   * @param[in]  username      Username
   * @param[in]  cert_file     The cert file
   * @param[in]  key_file      The key file
   *
   * @return     Session handle (free with close_session)
   */
  virtual session_t open_cert_session(const std::string& cipher_suite,
                                      const std::string& server_ip,
                                      const int server_port,
                                      const std::string& username,
                                      const std::string& cert_file,
                                      const std::string& key_file) = 0;

  /**
   * @brief      Server-side accept shared-key based session
   *
   * @param[in]  port            Network port
   * @param[in]  get_shared_key  Function to validate keys
   *
   * @return     Session handle (free with close_session)
   */
  virtual session_t accept_psk_session(const int port,
                                       std::function<const std::string(const std::string& username)> get_shared_key) = 0;

  /**
   * @brief      Client-side open shared-key based session
   *
   * @param[in]  server_ip    Server IP address
   * @param[in]  server_port  Server port
   * @param[in]  username     Username
   * @param[in]  key          Key
   *
   * @return     Session handle (free with close_session)
   */
  virtual session_t open_psk_session(const std::string& server_ip,
                                     const int server_port,
                                     const std::string& username,
                                     const std::string& key) = 0;

  /**
   * @brief      Export key material from existing session
   *
   * @param[in]  session   Session handle
   * @param[in]  label     Label used in PRF computation, typically a short string.
   * @param[in]  context   Optional extra data to seed the PRF with.
   * @param[in]  out_size  Size of result memory
   * @param[out] out_key   Result
   *
   * @return     S_OK on success
   */
  virtual status_t export_key(const session_t session,
                              const std::string& label,
                              const std::string& context,
                              const size_t out_size,
                              void * out_key) = 0;

  /**
   * @brief      Export key material from existing session
   *
   * @param[in]  session   Session handle
   * @param[in]  label     Label used in PRF computation, typically a short string.
   * @param[in]  context   Optional extra data to seed the PRF with.
   * @param[in]  out_size  Size of result memory
   * @param[out] out_key   Result
   *
   * @return     S_OK on success.
   */
  virtual status_t export_key(const session_t session,
                              const std::string& label,
                              const std::string& context,
                              const size_t out_size,
                              std::string& out_key) = 0;

  /**
   * @brief      Perform hashed MAC (Message Authentication Code)
   *
   * @param[in]  algo          Algorithm
   * @param[in]  key           Key
   * @param[in]  in_data       In data to perform MAC on
   * @param[in]  in_data_size  Size of data in bytes
   * @param      out_digest    Output digest
   *
   * @return     S_OK on success
   */
  virtual status_t hmac(component::ICrypto::mac_algorithm_t algo,
                        const std::string& key,
                        const void * in_data,
                        const size_t in_data_size,
                        std::string& out_digest) = 0;

  /**
   * @brief      Initializes the cipher for the session
   *
   * @param[in]  session  Session handle
   * @param[in]  cipher   Cipher choice
   * @param[in]  key      Key
   *
   * @return     S_OK on success
   */
  virtual status_t initialize_cipher(const session_t session,
                                     const cipher_t cipher,
                                     const std::string key) = 0;

  /**
   * @brief      AEAD encrypt (initialize_cipher should be called a priori)
   *             Authenticated Encryption with Associated Data
   *
   * @param[in]  session              Session handle
   * @param[in]  nonce                The nonce
   * @param[in]  nonce_len            The nonce length
   * @param[in]  auth                 Additional data to authenticate
   * @param[in]  auth_len             Length of additional data
   * @param[in]  plain_text           Plain text
   * @param[in]  plain_text_len       Plain text length
   * @param      out_cipher_text      Out cipher text
   * @param      out_cipher_text_len  Out cipher text length
   *
   * @return     S_OK on success
   */
  virtual status_t aead_encrypt(const session_t session,
                                const void * nonce,
                                size_t nonce_len,
                                const void * auth,
                                size_t auth_len,
                                const void * plain_text,
                                size_t plain_text_len,
                                void * out_cipher_text,
                                size_t * out_cipher_text_len) = 0;

  /**
   * @brief      AEAD decrypt (initialize_cipher should be called a priori)
   *             Authenticated Encryption with Associated Data
   *
   * @param[in]  session             Session handle
   * @param[in]  nonce               The nonce
   * @param[in]  nonce_len           The nonce length
   * @param[in]  auth                Additional data to authenticate
   * @param[in]  auth_len            Length of additional data
   * @param[in]  cipher_text         Cipher text
   * @param[in]  cipher_text_len     Cipher text length
   * @param      out_plain_text      Out plain text
   * @param      out_plain_text_len  Out plain text length
   *
   * @return     S_OK on success
   */
  virtual status_t aead_decrypt(const session_t session,
                                const void * nonce,
                                size_t nonce_len,
                                const void * auth,
                                size_t auth_len,
                                const void * cipher_text,
                                size_t cipher_text_len,
                                void * out_plain_text,
                                size_t * out_plain_text_len) = 0;

  /**
   * @brief      Send data as encrypted record
   *
   * @param[in]  session   Session handle
   * @param[in]  data      Data to send
   * @param[in]  data_len  Data length in bytes
   *
   * @return     Number of bytes sent or < 0 for error
   */
  virtual ssize_t record_send(const session_t session,
                              const void * data,
                              size_t data_len) = 0;

  /**
   * @brief      Receive data as encrypted record
   *
   * @param[in]  session   Session handle
   * @param[in]  data      Data placeholder
   * @param[in]  data_len  Data length in bytes
   *
   * @return     Number of bytes received or < 0 for error
   */
  virtual ssize_t record_recv(const session_t session,
                              void * data,
                              size_t data_len) = 0;
  /**
   * @brief      Closes a session.
   *
   * @param[in]  session  Session handle
   *
   * @return     S_OK on success
   */
  virtual status_t close_session(const session_t session) = 0;
};

class ICrypto_factory : public component::IBase {
 public:
  DECLARE_INTERFACE_UUID(0xFAC80DD6,0xBB71,0x48C3,0xAF6E,0x2C,0x2E,0xFD,0x7A,0x3B,0xD7);

  virtual ICrypto* create(unsigned debug_level, /* e.g., set TLS log level */
                          std::map<std::string, std::string>& params) = 0;
};
#pragma GCC diagnostic pop

}  // namespace component

#endif
