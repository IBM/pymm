#include <sys/mman.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <string.h>
#include <unistd.h>
#include <common/exceptions.h>
#include <common/net.h>
#include <regex>
namespace common
{

std::string get_eth_device_from_ip(const std::string& ipaddr)
{
  struct ifaddrs *ifaddr = nullptr, *ifa;
  int             s;
  char            host[NI_MAXHOST];
  std::string     result;

  if (getifaddrs(&ifaddr) == -1) throw General_exception("getifaddrs failed unexpectedly");

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;

    s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in), host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST);

    if (s == 0 && ipaddr == std::string(host)) {
      result = ifa->ifa_name;
      break;
    }
  }
  freeifaddrs(ifaddr);

  return result;
}

std::string get_ip_from_eth_device(const std::string& eth_device)
{
  struct ifaddrs *ifaddr, *ifa;
  int s;
  char host[NI_MAXHOST];
  std::string result;

  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");        
    throw General_exception("getifaddrs() failed");
  }

  for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == NULL)
        continue;  

      s = getnameinfo(ifa->ifa_addr,
                      sizeof(struct sockaddr_in),
                      host,
                      NI_MAXHOST,
                      NULL, 0, NI_NUMERICHOST);

      if((strcmp(ifa->ifa_name,eth_device.c_str())==0) && (ifa->ifa_addr->sa_family==AF_INET)) {
          if (s != 0)
            throw General_exception("getnameinfo() failed: %s\n", gai_strerror(s));
          result = host;
          break;
          // printf("\tInterface : <%s>\n",ifa->ifa_name );
          // printf("\t  Address : <%s>\n", host); 
        }
    }

  freeifaddrs(ifaddr);
  return result;
}


static std::string exec(const std::string cmd)
{
  std::array<char, 128>                    buffer;
  std::string                              result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

std::string get_eth_device_from_rdma_device(const std::string& rdma_device)
{
  auto exec_output = exec("ibdev2netdev | grep " + rdma_device);

  /*
    e.g.  mlx5_0 port 1 ==> enp5s0f0 (Up)
  */
  std::regex regex("\\s*\\S+ port [0-9] ==> (\\S+) \\(Up\\)");

  std::smatch m;
  std::regex_search(exec_output, m, regex);

  if (m.size() != 2) {
    PWRN("ibdev2netdev command failed or unable to parse output");
    throw General_exception("get_eth_device failed");
  }

  return m.str(1);
}


} // common
