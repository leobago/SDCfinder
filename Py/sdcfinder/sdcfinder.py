import numpy as np
import os
import cupy
import time
import json
import datetime
from .sysinfo import get_sys_info
from cupy.cuda import runtime

interval_sleep = 120
path_root_logs = "./"  # TODO: make this a parameter
metadata = {}
percent_memory_use = 0.5
batch_size = 1024 * 1024


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


def check_batch(batch):
    idx_nonzero = cupy.nonzero(batch)[0]
    # nonzero = batch[idx_nonzero]
    # nonzero_cpu = cupy.asnumpy(nonzero)
    # bits = np.unpackbits(nonzero_cpu)
    return idx_nonzero
    # print(bits)


def check_arrray(x, path):
    checksum = x.sum()
    print("checksum:", checksum)
    if checksum == 0:
        return
    print("detected {} currupted bytes".format(checksum))
    print("the following bytes have changed")
    cnt_batches = x.shape[0] // batch_size
    for i in range(cnt_batches):
        nonzero = check_batch(x[i * batch_size: i * batch_size + batch_size])
        if nonzero.shape[0] > 0:
            print("nonzero elements:", nonzero + i * batch_size)
            with open(os.path.join(path, "corruption.log"), "a") as myfile:
                print("nonzero elements:", nonzero + i * batch_size, file=myfile)
    x.fill(0)


def get_gpu_mem_size(id_device=0):
    mem_free, mem_total = runtime.memGetInfo()
    return mem_free


def run():
    path_out = os.path.join(path_root_logs, get_time_str())
    metadata["platform"] = get_sys_info()
    print(metadata)
    save_data_json(metadata, os.path.join(path_out, "metadata.json"))
    size = int(get_gpu_mem_size() * percent_memory_use)
    print("using {} B of memory".format(size))
    x = cupy.zeros(size, dtype=cupy.uint8)
    # x[123] = 7
    while True:
        time.sleep(interval_sleep)
        check_arrray(x, path_out)
