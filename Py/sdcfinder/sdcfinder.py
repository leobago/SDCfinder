import numpy as np
import cupy
import time
from .sysinfo import get_sys_info


interval_sleep = 3
path_logs = "/work/alex/data/memcheck"  # TODO: make this a parameter
metadata = {}


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
    metadata["platform"] = get_sys_info()
    print(metadata)
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
