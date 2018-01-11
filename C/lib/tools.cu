#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "MemoryReliability_decl.cuh"

#ifdef __cplusplus
extern "C" {
#endif
#include "logging.h"
#ifdef __cplusplus
};
#endif

__device__ __managed__ unsigned int fault_masking = 0;
__device__ __managed__ uintptr_t* faulting_addr;
__device__ __managed__ uintptr_t faulting_value;

__device__
bool is_first_error()
{
   return !atomicCAS(&fault_masking, 0, 1);
}

__host__
bool report_error()
{
   return (fault_masking == 1) ? true : false;
}

__global__
void check_gpu_kernel(uintptr_t* buffer,
                      const long unsigned num_words,
                      uintptr_t expected_value,
                      uintptr_t new_value)
{
    for (long unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
                       word < num_words;
                       word += gridDim.x * blockDim.x)
    {
        uintptr_t actual_value = buffer[word];
        if (actual_value != expected_value)
        {
            // Check if errors are masked:
            if (is_first_error())
            {
                faulting_addr = &buffer[word];
                faulting_value = actual_value;
            }
        }

        buffer[word] = new_value;
    }
}

void check_gpu_stub(uintptr_t* buffer,
                    unsigned long long num_bytes,
                    uintptr_t expected_value,
                    uintptr_t new_value)
{
    // We only report the first error, the rest are ignored
    fault_masking = 0;

    const long unsigned num_words = num_bytes / sizeof(uintptr_t);
    const int block_size = 128;
    const int grid_size = (num_words + block_size - 1) / block_size;
    check_gpu_kernel<<<grid_size, block_size>>>
            (buffer, num_words, expected_value, new_value);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        char msg[255];
        sprintf(msg, "ERROR_INFO, cudaDeviceSynchronize reported an error: (%s:%s).",
                      cudaGetErrorName(err), cudaGetErrorString(err));
        log_message(msg);
    }

    if (report_error())
        log_error(faulting_addr, faulting_value, expected_value);
}
