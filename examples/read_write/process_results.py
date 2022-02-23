import time
import os
import numpy as np
import os.path
import os
import argparse
import json
import statistics

def initial_output_arrays(keys, raw_data, proccessed):
    for i in keys:
        raw_data[i] = { "open_time": [], "copy_time" : [], "persist_time" : [], "close_time" : []}
        proccessed[i] = {"open_time" : {"avg": 0, "min": 0, "max" : 0, "std" : 0},
                         "copy_time" : {"avg": 0, "min": 0, "max" : 0, "std" : 0},
                         "persist_time" : {"avg": 0, "min": 0, "max" : 0, "std" : 0},
                         "close_time" : {"avg": 0, "min": 0, "max" : 0, "std" : 0},
                         "items" : 0}

def add_raw_data(results, raw_data):
    for i in results.keys():
        raw_data[i]["open_time"].append(results[i]["open_time"])
        raw_data[i]["copy_time"].append(results[i]["copy_time"])
        raw_data[i]["persist_time"].append(results[i]["persist_time"])
        raw_data[i]["close_time"].append(results[i]["close_time"])


def proccessed_data (raw_data, proccessed):
    for i in proccessed.keys():
        proccessed[i]["items"] = len(raw_data[i]["copy_time"])
        for j in ("open_time", "copy_time", "persist_time", "close_time"):
            proccessed[i][j]["avg"] = sum(raw_data[i][j])/ len(raw_data[i][j])
            proccessed[i][j]["max"] = max(raw_data[i][j])
            proccessed[i][j]["min"] = min(raw_data[i][j])
            proccessed[i][j]["std"] = statistics.stdev(raw_data[i][j])
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help=" The path to the input directory")
    parser.add_argument("start_index", type=str, help="The start file")
    parser.add_argument("end_index", type=str, help="The end file")
    parser.add_argument("benchmark_size", type=str, help="The benchmark size --- The file that we start is input_path/start_index_results.benchmark_size.results.json")
    parser.add_argument("output path", type=str, help="The path to the output directory")
    args = parser.parse_args()

    # proccess all the files
    raw_data = {}
    proccessed = {} 
    items = 0
    for i in range(int(args.start_index), int(args.end_index)+1):
        filename = args.input_path + "/" + str(i) + "_results." + args.benchmark_size + ".json"
        print (filename)
        with open(filename, 'r') as f:
            results = json.load(f)
            if (items == 0):
                initial_output_arrays(results.keys(), raw_data, proccessed)
            add_raw_data(results, raw_data)
            print (raw_data)
            items += 1
    proccessed_data (raw_data, proccessed)
    for j in ("open_time", "close_time", "persist_time", "copy_time"):
        print (j)
        for i in proccessed.keys():
            print("{0} {1}".format(i, proccessed[i][j]))         

#    results = run_tests_func(args)
#    with open(args.output_dir + "_results." + str(array_size_gb) + "GB.json" ,'w') as fp:
#            json.dump(results, fp)
#    print(json.dumps(results, indent=4, sort_keys=True))


if __name__ == "__main__":
            main()

# save (func_name, path, is_input_DRAM, time_create, copy_time, persist_time, close_time)  
