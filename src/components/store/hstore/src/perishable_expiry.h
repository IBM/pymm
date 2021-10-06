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


#ifndef MCAS_HSTORE_PERISHABLE_EXPIRY_H
#define MCAS_HSTORE_PERISHABLE_EXPIRY_H

#include "hop_hash_log.h"
#include <stdexcept>

struct perishable_expiry
	: public std::runtime_error
{
private:
	int _line;
	static unsigned _instance_count;
public:
	perishable_expiry(int line)
		: std::runtime_error("perishable timer expired")
		, _line(line)
	{
		++_instance_count;
		hop_hash_log<true>::write(LOG_LOCATION, "perishable count ", _instance_count);
	}
	perishable_expiry(const perishable_expiry &other)
		: std::runtime_error(other.what())
		, _line(other._line)
	{
		++_instance_count;
		hop_hash_log<true>::write(LOG_LOCATION, "perishable count ", _instance_count);
	}
	perishable_expiry(perishable_expiry &&other)
		: std::runtime_error(other.what())
		, _line(other._line)
	{
		++_instance_count;
		hop_hash_log<true>::write(LOG_LOCATION, "perishable count ", _instance_count);
	}
	~perishable_expiry()
	{
		--_instance_count;
		hop_hash_log<true>::write(LOG_LOCATION, "perishable count ", _instance_count);
	}
	/* true iff the current exception object exist and is a perishable_expiry */
	static bool is_current();
};

#endif
