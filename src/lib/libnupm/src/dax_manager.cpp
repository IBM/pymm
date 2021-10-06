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

#include "dax_manager.h"

#include "arena.h"
#include "arena_dev.h"
#include "arena_fs.h"
#include "arena_none.h"
#include "dax_data.h" /* DM_region_header */
#include "filesystem.h"
#include "nd_utils.h"

#include <common/exceptions.h>
#include <common/fd_locked.h>
#include <common/memory_mapped.h>
#include <common/utils.h>

#include <fcntl.h>
#include <sys/mman.h> /* MAP_LOCKED */
#include <boost/icl/split_interval_map.hpp>
#include <cinttypes>
#include <fstream>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

#define DEBUG_PREFIX "dax_manager: "

#if _NUPM_FILESYSTEM_STD_
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

namespace
{
	std::set<std::string> nupm_dax_manager_mapped;
	std::mutex nupm_dax_manager_mapped_lock;

	bool init_have_odp()
	{
		/* env variable USE_ODP to indicate On Demand Paging */
		char* p = ::getenv("USE_ODP");
		bool odp = true;
		if ( p != nullptr )
		{
			errno = 0;
			odp = bool(std::strtoul(p,nullptr,0));

			auto e = errno;
			if ( e == 0 )
			{
				PLOG("USE_ODP=%d (%s on-demand paging)", int(odp), odp ? "using" : "not using");
			}
			else
			{
				PLOG("USE_ODP specification %s failed to parse: %s", p, ::strerror(e));
			}
		}
		return odp;
	}

	int init_map_lock_mask()
	{
		/* On Demand Paging iimplies that mapped memory need not be pinned */
		return nupm::dax_manager::have_odp ? 0 : MAP_LOCKED;
	}
}

const bool nupm::dax_manager::have_odp = init_have_odp();
const int nupm::dax_manager::effective_map_locked = init_map_lock_mask();
constexpr const char *nupm::dax_manager::_cname;

nupm::path_use::path_use(path_use &&other_) noexcept
  : common::log_source(other_)
  , _name()
{
  using std::swap;
  swap(_name, other_._name);
}

nupm::path_use::path_use(const common::log_source &ls_, const string_view &name_)
  : common::log_source(ls_)
  , _name(name_)
{
  std::lock_guard<std::mutex> g(nupm_dax_manager_mapped_lock);
  bool inserted = nupm_dax_manager_mapped.insert(std::string(name_)).second;
  if ( ! inserted )
  {
    std::ostringstream o;
    o << __func__ << ": instance already managing path (" << name_ << ")";
    throw std::range_error(o.str());
  }
  CPLOG(3, "%s (%p): name: %s", __func__, common::p_fmt(this), _name.c_str());
}

nupm::path_use::~path_use()
{
  if ( _name.size() )
  {
    std::lock_guard<std::mutex> g(nupm_dax_manager_mapped_lock);
    nupm_dax_manager_mapped.erase(_name);
    CPLOG(3, "%s: dax mgr instance: %s", __func__, _name.c_str());
  }
}

std::pair<std::vector<common::byte_span>, std::size_t> get_mapping(const fs::path &path_map)
{
	/* A region must always be mapped to the same address, as MCAS
	 * MCAS software uses absolute addresses. Current design is to
	 * save this in a file extended attribute, ahtough it could be
	 * saved in a specially-named file.
	 */
	std::vector<common::byte_span> m;
	std::ifstream f(path_map.c_str());
	std::size_t covered = 0;
	std::uint64_t addr;
	std::size_t size;
	f >> addr >> size;
	while ( f.good() )
	{
		m.push_back(common::make_byte_span(reinterpret_cast<common::byte *>(addr), size));
		covered += size;
#if 0
		PLOG("%s %s: %p, 0x%zx", __func__, path_map.c_str(), common::p_fmt(m.back().data()), m.back().size());
#endif
		f >> addr >> size;
	}
	return { m, covered };
}

std::vector<common::byte_span> get_mapping(const fs::path &path_map, const std::size_t expected_size)
{
	auto r = get_mapping(path_map);
    if ( r.second != expected_size )
	{
		std::ostringstream o;
		o << __func__ << ": map file " << path_map << std::hex << std::showbase << " expected to cover " << expected_size << " bytes, but covers " << r.second << " bytes";
		throw std::runtime_error(o.str());
	}
	return r.first;
}

void nupm::dax_manager::data_map_remove(const fs::directory_entry &e, const std::string &)
{
	if (
#if _NUPM_FILESYSTEM_STD_
		e.is_regular_file()
#else
		fs::is_regular_file(e.status())
#endif
	)
	{
		auto p = e.path();
		static std::set<std::string> used_extensions { ".map", ".data" };
		if ( used_extensions.count(p.extension().string()) != 0 )
		{
			CPLOG(1, "%s remove %s", __func__, p.c_str());
			std::error_code ec;
			fs::remove(p, ec);
			if ( ec.value() == 0 )
			{
				PLOG("%s: removing %s: %s", __func__, p.c_str(), ec.message().c_str());
			}
		}
	}
	else if (
#if _NUPM_FILESYSTEM_STD_
		e.is_directory()
#else
		fs::is_directory(e.status())
#endif
	)
	{
		auto p = e.path();
		std::error_code ec;
		fs::remove(p, ec);
		if ( ec.value() == 0 )
		{
			CPLOG(2, "%s: removed %s: %s", __func__, p.c_str(), ec.message().c_str());
		}
	}
}

template <typename ...Args>
	int open_successful(const char *fn, int oflag, Args... args)
	{
		auto fd = ::open(fn, oflag, args...);
		if ( fd < 0 )
		{
			auto e = errno;
			throw std::system_error(std::error_code{e, std::system_category()}, std::string("opening ") + fn);
		}
		return fd;
	}

void nupm::dax_manager::map_register(const fs::directory_entry &de, const std::string &origin)
{
	if (
#if _NUPM_FILESYSTEM_STD_
		de.is_regular_file()
#else
		fs::is_regular_file(de.status())
#endif
	)
	{
		auto p = de.path();
		if ( p.extension().string() == ".data" )
		{
			CPLOG(1, "%s %s", __func__, p.c_str());

			auto pd = p;
			auto pm = p;
			pm.replace_extension(".map");
			p.replace_extension();
			auto id = p.string();
			id.erase(0, origin.size()+1); /* Assumes 1-character directory separator */
			/* first: mapping. second: mapping size */
			auto r = arena_fs::get_mapping(pm);

			/* NOT CHECKED: if mapping size not equal data file size, there is an inconsistency */

			/* insert and map the file */
			try
			{
				auto itb =
					_mapped_spaces.insert(
						mapped_spaces::value_type(
							id
							, space_registered(*this, this, common::fd_locked(open_successful(pd.c_str(), O_RDWR, 0666)), id, r.first)
						)
					);
				if ( ! itb.second )
				{
					throw std::runtime_error("multiple instances of path in configuration");
				}
				CPLOG(1, "%s: region %s at %p", __func__, itb.first->first.c_str(), ::base(itb.first->second._or.range()[0]));
			}
			catch ( const std::exception &e )
			{
				throw std::runtime_error("failed to register " + pd.string() + " :" + e.what());
			}
		}
	}
}

void nupm::dax_manager::files_scan(const path &p, const std::string &origin, void (dax_manager::*action)(const directory_entry &, const std::string &))
{
	std::error_code ec;
	auto ir = fs::directory_iterator(p, fs::directory_options::skip_permission_denied, ec);
	if ( ec.value() == 0 )
	{
		for ( auto e : ir )
		{
			if (
#if _NUPM_FILESYSTEM_STD_
				e.is_directory()
#else
				fs::is_directory(e.status())
#endif
			)
			{
				files_scan(e.path(), origin, action);
			}
			(this->*action)(e, origin);
		}
	}
}

std::unique_ptr<arena> nupm::dax_manager::make_arena_fs(
	const path &p
	, addr_t // base
	, bool force_reset
)
{
	if ( ! have_odp )
	{
		PWRN("%s arena %s is a directory but On Demand Paging is disabled. Run with USE_ODP=1 to enable ODP", __func__, p.c_str());
	}
	/* No checking. Although specifying a path twice would be odd, it causes no harm.
	 * But perhaps we will scan all address maps to develop a free address interval set.
	 */
	/* For all map files in the path, add covered addresses to _address_coverage and remove from
	 * _address_fs_available
	 */
	files_scan(p, p.string(), force_reset ? &dax_manager::data_map_remove : &dax_manager::map_register);
	return
		std::make_unique<arena_fs>(
			static_cast<log_source &>(*this)
			, p
		);
}

std::unique_ptr<arena> nupm::dax_manager::make_arena_none(
	const path &p
	, addr_t // base
	, bool // force_reset
)
{
	PLOG("%s: %s is unsuitable as an arena: neither a character file nor a directory", __func__, p.c_str());
	return
		std::make_unique<arena_none>(
			static_cast<log_source &>(*this)
			, p
		);
}

std::unique_ptr<arena> nupm::dax_manager::make_arena_dev(const path &p, addr_t base_, bool force_reset)
{
	/* Create and insert a space_registered.
	 *   path_use : tracks usage of the path name to ensure no duplicate uses
	 *   space_opened : tracks opened file descriptors, and the iov each represents
	 *   range_use : tracks vitutal address ranges to ensure no duplicate addresses
	 */
	auto id = p.string();
	try
	{
		auto itb =
			_mapped_spaces.insert(
				mapped_spaces::value_type(
					id
					, space_registered(*this, this, common::fd_locked(open_successful(p.c_str(), O_RDWR, 0666)), id, base_)
				)
			);
		if ( ! itb.second )
		{
			throw std::runtime_error("multiple instances of path " + p.string() + " in configuration");
		}
		CPLOG(0, "%s: region %s at %p", __func__, itb.first->first.c_str(), ::base(itb.first->second._or.range()[0]));
		return
			std::make_unique<arena_dev>(
				static_cast<log_source &>(*this)
				, id
				, recover_metadata(
					itb.first->second._or.range()[0],
					force_reset
				)
			);
	}
	catch ( const std::exception &e )
	{
		throw std::runtime_error("failed to make an arena for " + p.string() + " :" + e.what());
	}
}

bool nupm::dax_manager::enter(
	common::fd_locked && fd_
	, const string_view & id_
	, const std::vector<byte_span> &m_
)
{
	auto itb =
		_mapped_spaces.insert(
			mapped_spaces::value_type(
				std::string(id_)
				, space_registered(*this, this, std::move(fd_), id_, m_)
			)
		);
	if ( ! itb.second )
	{
		PLOG("%s: failed to insert %.*s (duplicate instance?)", __func__, int(id_.size()), id_.begin());
	}
	CPLOG(1, "%s: region %s at %p", __func__, itb.first->first.c_str(), ::base(itb.first->second._or.range()[0]));
	return itb.second;
}

void nupm::dax_manager::remove(const string_view & id_)
{
	auto itb = _mapped_spaces.find(std::string(id_));
	if ( itb != _mapped_spaces.end() )
	{
		CPLOG(2, "%s: _mapped_spaces found %.*s at %p", __func__, int(id_.size()), id_.begin(), common::p_fmt(&itb->second));
		CPLOG(1, "%s: region %s at %p", __func__, itb->first.c_str(), ::base(itb->second._or.range()[0]));
		_mapped_spaces.erase(itb);
	}
	else
	{
		CPLOG(2, "%s: _mapped_spaces does not contain %.*s", __func__, int(id_.size()), id_.begin());
	}
}

namespace nupm
{

dax_manager::dax_manager(
  const common::log_source &ls_,
  const std::vector<config_t>& dax_configs,
  bool force_reset
	, common::byte_span address_span_
)
  : common::log_source(ls_)
  , _nd()
  , _address_coverage()
  , _address_fs_available()
  /* space mapped by devdax */
  , _mapped_spaces()
  , _arenas()
  , _reentrant_lock()
{
  /* Maximum expected need is about 6 TiB (12 512GiB DIMMs).
   * Start, arbitrarily, at 0x10000000000
   */
	if ( ::data(address_span_) == nullptr )
	{
		address_span_ = common::make_byte_span(reinterpret_cast<byte *>(uintptr_t(1) << 40), (std::size_t(1) << 40));
	}
  auto i = boost::icl::interval<byte *>::right_open(::data(address_span_), ::data_end(address_span_));
  _address_fs_available.insert(i);

  /* set up each configuration */
  for ( const auto& config: dax_configs ) {

    CPLOG(0, DEBUG_PREFIX "region (%s,%lx)", config.path.c_str(), config.addr);

    /* a dax_config entry may be either devdax or fsdax.
     * If the path names a directory it is fsdax, else is it devdax.
     *
     * The startup behavior of devdax paths controlled by _mapped_spaces is:
     *   space_opened opens the path and maps the resulting fd
     * The shutdown behavior of devdax controlled by _mapped_spaces is:
     *   path_use calls nupm_dax_manager_mapped.erase(_path), a registry if files opened by this process
     *
     * The startup behavior of fsdax paths controlled by _mapped_spaces is:
     *   space_opened (None. Mapping are not attempted until open_region or create_region)
     * The shutdown behavior of devdax controlled by _mapped_spaces is:
     *   (None. Files are not opened until open_region or create_region)
     */

    path p(config.path);

    auto arena_make =
      fs::is_character_file(p) ? &dax_manager::make_arena_dev
      : fs::is_directory(p) ? &dax_manager::make_arena_fs
      : &dax_manager::make_arena_none;

    _arenas.emplace_back((this->*arena_make)(p, config.addr, force_reset));
  }
}

dax_manager::~dax_manager()
{
  CPLOG(0, "%s::%s", _cname, __func__);
}

void * dax_manager::locate_free_address_range(std::size_t size_)
{
	for ( auto i : _address_fs_available )
	{
		if ( ptrdiff_t(size_) <= i.upper() - i.lower() )
		{
			return i.lower();
		}
	}

	for ( auto i : _address_fs_available )
	{
		PLOG("%s: free %p..%p", __func__, common::p_fmt(i.lower()), common::p_fmt(i.upper()));
	}

	throw std::runtime_error(__func__ + std::string(" out of address ranges"));
}

auto dax_manager::lookup_arena(arena_id_t arena_id) -> arena *
{
  if ( _arenas.size() <= arena_id )
  {
    throw Logic_exception("%s::%s: could not find header for region (%d)", _cname, __func__,
                        arena_id);
  }
  return _arenas[arena_id].get();
}

void dax_manager::debug_dump(arena_id_t arena_id)
{
  guard_t g(_reentrant_lock);
  auto it = lookup_arena(arena_id);
  it->debug_dump();
}

auto dax_manager::open_region(
  const string_view & name
  , unsigned arena_id
) -> region_descriptor
{
  guard_t           g(_reentrant_lock);
  return lookup_arena(arena_id)->region_get(name);
}

auto dax_manager::create_region(
  const string_view &name_
  , arena_id_t arena_id_
  , const size_t size_
) -> region_descriptor
{
  guard_t           g(_reentrant_lock);
  auto arena = lookup_arena(arena_id_);
  CPLOG(1, "%s: %s size %zu arena_id %u", __func__, name_.begin(), size_, arena_id_);
  try
  {
    auto r = arena->region_create(name_, this, size_);
    CPLOG(0, "%s: path %s id  %.*s size req 0x%zx created at %p:%zx", __func__, arena->describe().data(), int(name_.size()), name_.begin(),  size_, ::base(r.address_map()[0]), ::size(r.address_map()[0]));
    return r;
  }
  catch ( const General_exception &e )
  {
    CPLOG(2,"%s: path %s id %.*s size req 0x%zx create failed (available 0x%zx) %s", __func__, arena->describe().data(), int(name_.size()), name_.begin(), size_, arena->get_max_available(), e.cause());
    return region_descriptor();
  }
  catch ( const std::exception &e )
  {
    CPLOG(2,"%s: path %s id %.*s size req 0x%zx create failed (available 0x%zx) %s", __func__, arena->describe().data(), int(name_.size()), name_.begin(), size_, arena->get_max_available(), e.what());
    return region_descriptor();
  }
}

auto dax_manager::resize_region(
  const string_view & id_
  , const arena_id_t arena_id_
  , const size_t size_
) -> region_descriptor
{
  guard_t           g(_reentrant_lock);
  auto arena = lookup_arena(arena_id_);
  CPLOG(1, "%s: %.*s size %zu", __func__, int(id_.size()), id_.begin(), size_);
  auto it = _mapped_spaces.find(std::string(id_));
  if ( it == _mapped_spaces.end() )
  {
    PLOG("%s: failed to find %.*s", __func__, int(id_.size()), id_.begin());
    throw std::runtime_error(std::string(__func__) + ": failed to find " + std::string(id_));
  }
  else
  {
    arena->region_resize(&it->second, size_);
  }
  return arena->region_get(id_);
}

void dax_manager::erase_region(const string_view & name, arena_id_t arena_id)
{
  guard_t           g(_reentrant_lock);
  lookup_arena(arena_id)->region_erase(name, this);
}

std::list<std::string> dax_manager::names_list(arena_id_t arena_id)
{
	guard_t g(_reentrant_lock);
	return lookup_arena(arena_id)->names_list();
}

size_t dax_manager::get_max_available(arena_id_t arena_id)
{
  guard_t           g(_reentrant_lock);
  return lookup_arena(arena_id)->get_max_available();
}

auto dax_manager::recover_metadata(const byte_span iov_,
                                      bool        force_rebuild) -> DM_region_header *
{
  assert(::base(iov_));
  DM_region_header *rh = static_cast<DM_region_header *>(::base(iov_));

  if ( force_rebuild || ! rh->check_magic() ) {
    PLOG("%s::%s: creating.", _cname, __func__);
    rh = new (::base(iov_)) DM_region_header(::size(iov_));
    PLOG("%s::%s: created.", _cname, __func__);
  }
  else {
    PLOG("%s::%s: recovering.", _cname, __func__);
    rh->check_undo_logs();
    PLOG("%s::%s: recovered.", _cname, __func__);
  }

  return rh;
}

}  // namespace nupm
