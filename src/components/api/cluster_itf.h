/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef __API_CLUSTER_ITF__
#define __API_CLUSTER_ITF__

#include <functional>
#include <common/exceptions.h>
#include <component/base.h>

namespace component
{

/**
 * Local IP-based clustering
 *
 */
class ICluster : public component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x9e9f55e3,0xf056,0x4273,0xa9c2,0x63,0xab,0x28,0xaa,0xa5,0x81);

  /**
   * Start node
   *
   */
  virtual void start_node() = 0;

  /**
   * Stop node; signal peers gracefully
   *
   */
  virtual void stop_node() = 0;

  /**
   * Ungraceful destroy node
   *
   */
  virtual void destroy_node() = 0;

  /**
   * Create/join a group
   *
   * @param group Name of group
   */
  virtual void group_join(const std::string& group) = 0;

  /**
   * Leave a group
   *
   * @param group Name of group
   */
  virtual void group_leave(const std::string& group) = 0;

  /**
   * Broadcast message to group
   *
   * @param group Group to send to
   * @param type Message type designator
   * @param message Message to send
   */
  virtual void shout(const std::string& group,
                     const std::string& type,
                     const std::string& message) = 0;

  /**
   * Send message to single peer
   *
   * @param peer_uuid Peer UUID
   * @param type Message type designator
   * @param message Message to send
   */
  virtual void whisper(const std::string& peer_uuid,
                       const std::string& type,
                       const std::string& message) = 0;

  /**
   * Poll for incoming messages
   *
   * @param sender_uuid Sennder UUID
   * @param type Message type
   * @param message Message content
   * @param values Vector of value strings (may be empty)
   *
   * @return True if message received
   */
  virtual bool poll_recv(std::string& sender_uuid,
			 std::string& type,
                         std::string& message,
			 std::vector<std::string>& values) = 0;




  enum class Timeout_type
    {
     EVASIVE,
     EXPIRED,
     SILENT,
    };

  /**
   * Set timeouts
   *
   * @param interval_ms
   */
  virtual status_t set_timeout(const Timeout_type type,
			       const int interval_ms) = 0;

  /**
   * Return the node name
   *
   *
   * @return
   */
  virtual std::string node_name() const = 0;

  /**
   * Return cluster-wide unique identifier for this node
   *
   *
   * @return UUID of the node
   */
  virtual std::string uuid() const = 0;

  /**
   * Dump debugging information
   *
   */
  virtual void dump_info() const = 0;

  /**
   * Convert UUID to peer address
   *
   * @param uuid
   *
   * @return
   */
  virtual std::string peer_address(const std::string& uuid) = 0;

};


class ICluster_factory : public component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacf55e3,0xf056,0x4273,0xa9c2,0x63,0xab,0x28,0xaa,0xa5,0x81);

  /**
   * @brief      Create cluster instance
   *
   * @param[in]  debug_level The debug level
   * @param[in]  node_name   The node name
   * @param[in]  nic         The nic
   * @param[in]  port        The network port
   *
   * @return     Reference to new instance of cluster
   */
  virtual ICluster * create(const unsigned debug_level,
                            const std::string& node_name,
                            const std::string& nic,
                            const unsigned int port) = 0;

};

} // component

#endif // __API_RDMA_ITF__
