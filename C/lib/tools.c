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

Filename    : tools.c
Authors     : Ferad Zyulkyarov, Kai Keller, Pau Farré, Leonardo Bautista-Gomez
Version     :
Copyright   :
Description : A daemon which tests the memory for errors.

This file provides helper functions.
*/

// TODO
// 
// the msr register name for ECC is IA32_MCi_STATUS. Explanations of the 
// fields are given in Intel® 64 and IA-32 Architectures Software 
// Developer’s Manual, Volume 3B: System Programming Guide, Part 2 in 
// chapter 15.3.2.1 (page 48 in the pdf). Position in msr file (in hex) 
// is 424H and for IA32_MC9_STATUS, where "Banks MC9 through MC 16 report 
// MC error from each channel of the integrated memory controllers."
//
// In the registers are also ecc counter.

#include "MemoryReliability_decl.h"

bool is_initialized(void* ptr)
{
    return ptr != NULL;
}

void initialize_cpu_memory()
{
    if (!is_initialized(cpu_mem))
    {
        cpu_mem = malloc(NumBytesCPU);
        if (cpu_mem == NULL)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot allocate %llu number of CPU memory.", NumBytesCPU);
            log_message(msg);
            sleep(2);
            if (daemon_pid_file_exists())
            {
                daemon_pid_delete_file();
            }
            exit(EXIT_FAILURE);
        }
        memset(cpu_mem, 0x0, NumBytesCPU);
    }
}

void initialize_gpu_memory()
{
#ifdef USE_CUDA
    if (!is_initialized(gpu_mem))
    {
        cudaError_t err;
        err = cudaMalloc(&gpu_mem, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot allocate %llu bytes of GPU memory (%s:%s).",
                    NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);

            goto fn_fatal_error;
        }
        err = cudaMemset(gpu_mem, 0x0, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot memset %llu bytes of GPU memory (%s:%s).",
                    NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);

            goto fn_fatal_error;
        }
    }
#endif
    return;

  fn_fatal_error:
    sleep(2);
    if (daemon_pid_file_exists())
    {
        daemon_pid_delete_file();
    }
    exit(EXIT_FAILURE);
}

void free_cpu_memory()
{
    free(cpu_mem);
}

void free_gpu_memory()
{
#ifdef USE_CUDA
    cudaError_t err;
    err = cudaFree(gpu_mem);
    if (err != cudaSuccess)
    {
        char msg[255];
        sprintf(msg, "ERROR_INFO,Cannot free GPU memory (%s:%s).",
                    cudaGetErrorName(err), cudaGetErrorString(err));
        log_message(msg);

        sleep(2);
        if (daemon_pid_file_exists())
        {
            daemon_pid_delete_file();
        }
        exit(EXIT_FAILURE);
    }
#endif
}

