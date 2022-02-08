#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/string_view.h>
#include <common/utils.h> /* MiB */
#include <gsl/pointers> /* not_null */

#include <cstddef> /* size_t */ 
#include <memory> /* unique_ptr */ 
#include <stdexcept> /* runtime_error */
#include <string>

namespace
{
	/* things which differ depending on the type of store used */
	struct custom_store
	{
		virtual ~custom_store() {}
		virtual std::size_t minimum_size(std::size_t s) const { return s; }
		virtual component::uuid_t factory() const = 0;
		virtual std::size_t presumed_allocation() const = 0;
		virtual bool uses_numa_nodes() const { return false; }
		virtual status_t rc_unknown_attribute() const { return E_NOT_SUPPORTED; }
		virtual status_t rc_percent_used() const { return S_OK; }
		virtual status_t rc_resize_locked() const { return E_LOCKED; }
		virtual status_t rc_attribute_key_null_ptr() const { return E_BAD_PARAM; }
		virtual status_t rc_attribute_key_not_found() const { return component::IKVStore::E_KEY_NOT_FOUND; }
		virtual status_t rc_attribute_hashtable_expansion() const { return S_OK; }
		virtual status_t rc_out_of_memory() const { return component::IKVStore::E_TOO_LARGE; }
		virtual status_t rc_atomic_update() const { return S_OK; }
		virtual status_t rc_allocate_pool_memory_size_0() const { return S_OK; }
		virtual bool swap_updates_timestamp() const { return false; }
	};

	struct custom_mapstore
		: public custom_store
	{
		virtual component::uuid_t factory() const override { return component::mapstore_factory; }
		std::size_t minimum_size(std::size_t s) const override { return std::max(std::size_t(8), s); }
		std::size_t presumed_allocation() const override { return 1ULL << DM_REGION_LOG_GRAIN_SIZE; }
		bool uses_numa_nodes() const override { return true; }
		status_t rc_unknown_attribute() const override { return E_INVALID_ARG; }
		status_t rc_percent_used() const override { return rc_unknown_attribute(); }
		status_t rc_resize_locked() const override { return E_INVAL; }
		status_t rc_attribute_key_null_ptr() const override { return E_INVALID_ARG; }
		status_t rc_attribute_key_not_found() const override { return rc_unknown_attribute(); }
		status_t rc_attribute_hashtable_expansion() const override { return rc_unknown_attribute(); }
		status_t rc_out_of_memory() const override { return E_INVAL; }
		status_t rc_atomic_update() const override { return E_NOT_SUPPORTED; }
		status_t rc_allocate_pool_memory_size_0() const override { return E_INVAL; }
	};

	struct custom_hstore
		: public custom_store
	{
		virtual component::uuid_t factory() const override { return component::hstore_factory; }
		std::size_t presumed_allocation() const override { return MiB(32); }
		bool swap_updates_timestamp() const override { return true; }
	};

	struct custom_hstore_cc
		: public custom_hstore
	{
		std::size_t presumed_allocation() const override { return MiB(32); }
	};

	custom_mapstore custom_mapstore_i{};
	custom_hstore custom_hstore_i{};
	custom_hstore_cc custom_hstore_cc_i{};

	const std::map<std::string, gsl::not_null<custom_store *>, std::less<>> custom_map =
	{
		{ "mapstore", &custom_mapstore_i },
		{ "hstore", &custom_hstore_i },
		{ "hstore-cc", &custom_hstore_cc_i },
		{ "hstore-mc", &custom_hstore_i },
		{ "hstore-mr", &custom_hstore_i },
		{ "hstore-mm", &custom_hstore_i },
	};
}

gsl::not_null<custom_store *> locate_custom_store(common::string_view store)
{
	const auto c_it = custom_map.find(store);
	if ( c_it == custom_map.end() )
    {
		throw std::runtime_error(common::format("store {} not recognized", store));
    }
	return c_it->second;
}

auto make_kvstore(
	common::string_view store
	, gsl::not_null<custom_store *> c
	, const component::IKVStore_factory::map_create & mc
) -> std::unique_ptr<component::IKVStore>
{
	using IKVStore_factory = component::IKVStore_factory;
	const std::string store_lib = "libcomponent-" + std::string(store) + ".so";
	auto *comp = component::load_component(store_lib.c_str(), c->factory());
	if ( ! comp )
    {
		throw std::runtime_error(common::format("failed to load component {}", store_lib));
    }

	const auto fact = make_itf_ref(static_cast<IKVStore_factory*>(comp->query_interface(IKVStore_factory::iid())));
	const auto kvstore = fact->create(0, mc);

	return std::unique_ptr<component::IKVStore>(kvstore);
}
