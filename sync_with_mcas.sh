#!/bin/bash
#
# IBM (c) 2022
#
# Script used to update PyMM repo *from* MCAS repo - be careful!
#
if [[ -z "${MCAS_SOURCE_HOME}" ]]; then
    echo "Environment variable MCAS_SOURCE_HOME is not set"
    exit -1
fi

echo "MCAS source: ${MCAS_SOURCE_HOME}"

rsync -rv ${MCAS_SOURCE_HOME}/src/components/api/ ./src/components/api/
rsync -rv ${MCAS_SOURCE_HOME}/src/components/store/ ./src/components/store/
rsync -rv ${MCAS_SOURCE_HOME}/src/mm/ ./src/mm/
rsync -rv ${MCAS_SOURCE_HOME}/src/python/pymm/ ./src/python/pymm/

for i in cityhash common EASTL googletest GSL libccpm libmm libnupm libpmem ndctl rapidjson;
do
    echo $i
    rsync -rv ${MCAS_SOURCE_HOME}/src/lib/$i/ ./src/lib/$i/
done
