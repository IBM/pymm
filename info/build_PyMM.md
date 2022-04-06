## How to build

Upadte submodels:
```bash
git submodule update --init --recursive
```

Run dependencies for your OS
```
cd deps
./install-<Your-OS-Version>.sh
./install-python-deps.sh
cd ../
```

Create Build directory:
```bash
mkdir build
cd build
```

Configure as debug build:
```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist .. 
```

Build bootstrap libraries:
```bash
make bootstrap
```

Build everything else (-j optional for parallel build):

```
make -j install
```
