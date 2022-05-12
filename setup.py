# Welcome to the PyMM setup.py.
#
#   Build Flags: 
#      Optimize[Default] / Debug
#      --debug     
#  

import subprocess
import os.path
import argparse
from os import path

def build_pymm(args):
    print ("Updating submodule:")
    print("git submodule update --init --recursive")
    os.system("git submodule update --init --recursive")
    if (not path.exists('build')):
        print("Directory build Created ")
        os.mkdir('build')
    else:
        print("Directory build already exists")
    
    print ("move to build directory")
    os.chdir('build')
   
    flags = 'Release'
    if (args.debug):
        flags = 'DEBUG'
    
    print("cmake -DCMAKE_BUILD_TYPE=" + flags + " -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..")
    command = "cmake -DCMAKE_BUILD_TYPE=" + flags + " -DCMAKE_INSTALL_PREFIX:PATH=" + os.getcwd() + "/dist .."
    print (command)
    os.system(command)
    print("make bootstrap")
    os.system("make bootstrap")
    
    print("make -j install")
    os.system("make -j install")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", default=False, help="Build in debug mode")
    args = parser.parse_args()
    build_pymm(args)


if __name__ == "__main__":
          main()
