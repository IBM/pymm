#include <api/components.h>
#include <api/kvstore_itf.h>
#include <limits.h>
#include <cstdlib>
#include <boost/program_options.hpp>
#include <cassert>
#include <common/json.h>
#include <common/logging.h>
#include <common/utils.h> /* MiB */
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>

using namespace component;
using namespace std;

namespace
{
	/* things which differ depending on the type of store used */
	struct custom_store
	{
		virtual ~custom_store() {}
		virtual std::size_t minimum_size(std::size_t s) const { return s; }
		virtual component::uuid_t factory() const = 0;
		virtual std::size_t presumed_allocation() const = 0;
	};

	struct custom_mapstore
		: public custom_store
	{
		virtual component::uuid_t factory() const override { return component::mapstore_factory; }
		std::size_t minimum_size(std::size_t s) const override { return std::max(std::size_t(8), s); }
		std::size_t presumed_allocation() const override { return 1ULL << DM_REGION_LOG_GRAIN_SIZE; }
	};

	struct custom_hstore
		: public custom_store
	{
		virtual component::uuid_t factory() const override { return component::hstore_factory; }
		std::size_t presumed_allocation() const override { return MiB(32); }
	};

	struct custom_hstore_cc
		: public custom_hstore
	{
		std::size_t presumed_allocation() const override { return MiB(32); }
	};

	custom_mapstore custom_mapstore_i{};
	custom_hstore custom_hstore_i{};
	custom_hstore_cc custom_hstore_cc_i{};
}

component::IKVStore* init(unsigned debug_level, const string& store, const custom_store &c, const string &daxconfig)
{
  string store_lib = "libcomponent-" + store + ".so";
  IBase* comp = load_component(store_lib.c_str(), c.factory());
  assert(comp);
  auto fact    = make_itf_ref(static_cast<IKVStore_factory*>(comp->query_interface(IKVStore_factory::iid())));
  auto kvstore = fact->create(
      debug_level, {{+component::IKVStore_factory::k_name, "numa0"},
          {+component::IKVStore_factory::k_dax_config, daxconfig}
         });

  return kvstore;
}


struct Options {
  string   daxconfig;
} g_options{};


int main(int argc, char* argv[])
try
{
  namespace po = boost::program_options;

  po::options_description            desc("Options");
  po::positional_options_description g_pos; /* no positional options */

  namespace c_json = common::json;
  using json = c_json::serializer<c_json::dummy_writer>;

  desc.add_options()
    ("help,h", "Show help")
#if 0
    ("aligned", "Whether to make random size 2^n aligned")
    (
      "exponential", "Size distribution should be expoenntial (not linear). Default true iff --aligned is specified")(
      "alignment", po::value<unsigned>()->default_value(8), "Alignment hint for allocate memory")(
      "iteration", po::value<unsigned>()->default_value(1), "Iterations to run the test")(
      "minsize", po::value<unsigned>()->default_value(1), "Min size to allocate")(
      "maxsize", po::value<unsigned>()->default_value(4096), "Max size to allocate")(
      "store", po::value<std::string>()->default_value("mapstore"), "Store type to test: e.g., hstore, hstore-cc, mapstore")
#endif
      ("daxconfig", po::value<std::string>()->default_value(
        json::array(
          json::object(
            json::member("path", "/dev/dax0.0")
            , json::member("addr", 0x9000000000)
          )
        , json::object(
            json::member("path", "/dev/dax1.0")
/* Overlapping and non-overlapping mappings, if size of /dev/dax0.0 is 0x5e7e00000 */
#if 1
            , json::member("addr", 0x95e0000000) /* overlapping, tests for overlap check */
#else
            , json::member("addr", 0x9600000000) /* okay */
#endif
          )
        ).str()
  )
);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(g_pos).run(), vm);

  if (vm.count("help") > 0) {
    std::cout << desc;
    return -1;
  }

  g_options.daxconfig = vm["daxconfig"].as<std::string>();

#if 0
using p = custom_store *;
using m = std::map<std::string, p>;
	const m custom_map =
	{
		{ "mapstore", &custom_mapstore_i },
		{ "hstore", &custom_hstore_i },
		{ "hstore-cc", &custom_hstore_cc_i },
	};

	const auto c_it = custom_map.find(g_options.store);
	if ( c_it == custom_map.end() )
    {
      PLOG("store %s not recognized", g_options.store.c_str());
    }
#endif
  try
  {
    auto kvstore = init(2, "hstore", custom_hstore_i, g_options.daxconfig);
    (void) kvstore;
    assert(0);
  }
  catch (std::domain_error &e)
  {
    return 0;
  }
}
catch ( const Exception &e )
{
  PERR("Exception %s", e.cause());
  return 1;
}
catch ( const std::exception &e )
{
  PERR("exception %s", e.what());
  return 1;
}
