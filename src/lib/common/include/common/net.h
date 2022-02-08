#ifndef __COMMON_NET_H__
#define __COMMON_NET_H__

#include <common/common.h>
#include <string>

namespace common
{

/**
 * Convert ip address to network interface name
 *
 * @param ipaddr IP address string
 *
 * @return Network interface name
 */
std::string get_eth_device_from_ip(const std::string& ipaddr);


/**
 * Get IP for ethernet device
 *
 * @param eth_device Network interface name
 *
 * @return IP address string
 */
std::string get_ip_from_eth_device(const std::string& eth_device);

/**
 * Get ethernet device corresponding to RDMA device
 *
 * @param rdma_device RDMA device (e.g., mlx5_0)
 *
 * @return Ethernet device string
 */
std::string get_eth_device_from_rdma_device(const std::string& rdma_device);
}

#endif
