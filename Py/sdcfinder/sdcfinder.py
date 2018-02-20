import numpy as np
import os
import cupy
import time
import json
import datetime
from .sysinfo import get_sys_info


interval_sleep = 3
path_root_logs = "/work/alex/data/memcheck"  # TODO: make this a parameter
metadata = {}


def get_time_str():
    d = datetime.datetime.now()
    s = d.strftime("%y.%m.%d_%H.%M.%S")
    return s


def save_data_json(data, name_file):
    path = os.path.realpath(os.path.dirname(name_file))
    os.makedirs(path, exist_ok=True)
    s = json.dumps(data, ensure_ascii=False, indent=4, sort_keys=True)
    f = open(name_file, 'w')
    print(s, file=f)
    f.close()


def analize(x):
    print("the following elements are nonzero")
    idx_nonzero = cupy.nonzero(x)
    nonzero = x[idx_nonzero]
    nonzero_cpu = cupy.asnumpy(nonzero)
    bits = np.unpackbits(nonzero_cpu)
    print(idx_nonzero)
    print(bits)


def get_gpu_mem_size(id_device=0):
    return 10050


def run():
    path_out = os.path.join(path_root_logs, get_time_str())
    metadata["platform"] = get_sys_info()
    print(metadata)
    save_data_json(metadata, os.path.join(path_out, "metadata.json"))
    size = get_gpu_mem_size()
    x = cupy.zeros(size, dtype=cupy.uint8)
    while True:
        time.sleep(interval_sleep)
        checksum = x.sum()
        print("checksum:", checksum)
        x[1] = 7
        if checksum != 0:
            print("detected curruption", checksum)
            analize(x)
