/*
 *
 * BSD 3-Clause License
 *
 * Copyright (c) 2017, Leo
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ============================================================================
*/

#include "MemoryReliability_decl.h"

bool check_cpu_mem(ADDRVALUE* buffer,
                   unsigned long long num_bytes,
                   ADDRVALUE expected_value,
                   ADDRVALUE new_value)
{
    const long unsigned num_words = num_bytes / sizeof(ADDRVALUE);
    for (long unsigned word = 0; word < num_words; word++)
    {
        ADDRVALUE actual_value = buffer[word];
        if (actual_value != expected_value)
        {
            log_error(&buffer[word], actual_value, expected_value);
        }

        buffer[word] = new_value;
    }
    return true;
}

bool check_gpu_mem(ADDRVALUE* buffer,
                   unsigned long long num_bytes,
                   ADDRVALUE expected_value,
                   ADDRVALUE new_value)
{
#if defined(USE_CUDA)
    check_gpu_stub(buffer, num_bytes, expected_value, new_value);
#endif
    return true;
}

void simple_memory_test()
{

    ADDRVALUE expected_value = 0;

    //
    // The main loop
    //
    while(ExitNow == 0)
    {
        check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, expected_value+1);
        check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, expected_value+1);

        expected_value++;

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }

    }

}

void zero_one_test()
{
    
    ADDRVALUE expected_value = 0x0;
    //
    // The main loop
    //
    while(ExitNow == 0)
    {
        //
        // Simulate error at random position and random iteration.
        //
        inject_cpu_error(cpu_mem, expected_value, ZERO);
        inject_gpu_error(gpu_mem, expected_value, ZERO);

        check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, ~expected_value);
        check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, ~expected_value);

        expected_value = ~expected_value;

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }

    }

}

void random_pattern_test()
{

    ADDRVALUE expected_value;
    memset(&expected_value, mem_pattern, sizeof(uint64_t));
    if (CheckCPU)
    {
        memset(cpu_mem, mem_pattern, NumBytesCPU);
    }
#if defined(INJECT_ERR) && defined(USE_CUDA)
    if (CheckGPU)
    {
        cudaError_t err = cudaMemset(gpu_mem, mem_pattern, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR,Cannot memset ptr_data to 0x1 (%s:%s).",
                         cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);
        }
    }
#endif

    //
    // The main loop
    //
    while (ExitNow == 0)
    {
        // 
        // Simulate error at random position and random iteration.
        //
        if (CheckCPU)
            inject_cpu_error(cpu_mem, expected_value, RANDOM);
        if (CheckGPU)
            inject_gpu_error(gpu_mem, expected_value, RANDOM);

        if (CheckCPU)
            check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, ~expected_value);
        if (CheckGPU)
            check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, ~expected_value);

        expected_value = ~expected_value;

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }
    }


}

void memory_test_loop(Strategy_t type)
{
    if (type == SIMPLE) {
        simple_memory_test();
    } else if (type == ZERO) {
        zero_one_test();
    } else if (type == RANDOM) {
        random_pattern_test();
    }
}

