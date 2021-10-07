#!/bin/bash

FILES=$(grep -lr '\W/home/' $@ | grep -v -E './src/lib/googletest|./src/lib/zyre|./src/components/ado/.*_proxy/unit_test')

if [[ -n "$FILES" ]]
then :
	cat <<EOF
Error: Apparent reference(s) to a /home directory in these filee:

$FILES

Git files should generally avoid references to /home directories.
EOF
	exit 1
fi

FILES=$(find $@ -type f ! -size 0 -exec grep -IL . "{}" \; | grep -v -E './.git|./tools|./deploy|./src/lib/(EASTL|flatbuffers|libbase64|rapidjson|tbb|zyre)')
if [[ -n "$FILES" ]]
then :
	cat <<EOF
Error: Binary files:

$FILES

Git files should generally contain source code, not binary data.
EOF
	exit 1
fi


