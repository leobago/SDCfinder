#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "MemoryReliability_decl.cuh"

#define MAX_GPU_ERRORS 100

#ifdef __cplusplus
extern "C" {
#endif
#include "logging.h"
#ifdef __cplusplus
};
#endif

typedef struct {
    const ADDRVALUE* addr;
    ADDRVALUE  value;
} error_info_t;

__device__ __managed__ error_info_t gpu_errors[MAX_GPU_ERRORS];
__device__ __managed__ int gpu_error_count;

__device__
void add_error(const ADDRVALUE* addr, const ADDRVALUE value)
{
    // inc before checking to avoid race conditions and to
    // keep the real error count
    unsigned int error_idx = atomicAdd(&gpu_error_count, 1);
    if (error_idx < MAX_GPU_ERRORS) {
        gpu_errors[error_idx].addr  = addr;
        gpu_errors[error_idx].value = value;
    }
}

__global__
void check_gpu_kernel(ADDRVALUE* buffer,
                      const long unsigned num_words,
                      ADDRVALUE expected_value,
                      ADDRVALUE new_value)
{
    for (long unsigned word = blockIdx.x * blockDim.x + threadIdx.x;
                       word < num_words;
                       word += gridDim.x * blockDim.x)
    {
        ADDRVALUE actual_value = buffer[word];
        if (actual_value != expected_value)
        {
            add_error(&buffer[word], actual_value);
        }

        buffer[word] = new_value;
    }
}

void check_gpu_stub(ADDRVALUE* buffer,
                    unsigned long long num_bytes,
                    ADDRVALUE expected_value,
                    ADDRVALUE new_value)
{
    gpu_error_count = 0;

    const long unsigned num_words = num_bytes / sizeof(ADDRVALUE);
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

    // Report up to MAX_GPU_ERRORS
    int report_count = (gpu_error_count > MAX_GPU_ERRORS) ? MAX_GPU_ERRORS : gpu_error_count;
    for (int i=0; i < report_count; i++)
    {
        log_error((void*)gpu_errors[i].addr, gpu_errors[i].value, expected_value);
    }
    if (gpu_error_count > MAX_GPU_ERRORS)
    {
        char msg[255];
        sprintf(msg, "ERROR_INFO,Only the first %d GPU errors were reported (total count is %d).", MAX_GPU_ERRORS, gpu_error_count);
        log_message(msg);
    }
}
