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

Filename    : MemoryReliability_decl.h
Authors     : Ferad Zyulkyarov, Kai Keller, Pau Farré, Leonardo Bautista-Gomez
Version     :
Copyright   :
Description : A daemon which tests the memory for errors.

Header file for MemoryReliability.c
*/

#ifndef _MEMORYRELIABILITY_DECL_H
#define _MEMORYRELIABILITY_DECL_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "MemoryReliability_decl.cuh"
#endif
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <errno.h>
#include <getopt.h>
#include <assert.h>

/** For ARM
#include "pmu.h"
*/

// DEFINITIONS

extern const unsigned int KILO;//                 = 1024;
extern const unsigned int MEGA;//                 = 1048576;
extern const unsigned int GIGA;//                 = 1073741824;

// TYPES

typedef uintptr_t ADDRVALUE;
typedef void* ADDRESS;

typedef enum Strategy {
    SIMPLE = 0,
    ZERO = 1,
    RANDOM = 2
} Strategy_t;

// STATIC VARIABLES

extern char PidFileName[255];// = "pid.txt";
extern char OutFile[255];// = "MemoryReliability.log";
extern char WarnFile[255];// = "MemoryReliability.warn";
extern char ErrFile[255];// = "MemoryReliability.err";
extern char DbgFile[255];// = "MemoryReliability.err";

extern unsigned int WarnRate;// = 0;
extern unsigned int NumErrors;// = 0;

char HostName[255];

unsigned long long NumBytesCPU;// = 0;
unsigned long long NumBytesGPU;// = 0;
void* cpu_mem;// = NULL;
void* gpu_mem;// = NULL;
unsigned char mem_pattern;

extern unsigned int SleepTime;// = 0;
extern bool ExitNow;// = 0;
extern int IsDaemonStart;// = 0;
extern int IsDaemonStop;// = 0;
bool CheckCPU;
bool CheckGPU;

// FUNCTIONS

void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);
void inject_gpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);
uintptr_t virtual_to_physical_address(uintptr_t virt_addr);
void memory_test_loop(Strategy_t type);
void log_message(char* message);
void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value);
void log_local_mem_errors(int local_mem_errors); // Logs local memory errors
bool parse_arguments(int argc, char* argv[]);
void print_usage(char* program_name);
void check_no_daemon_running();
void initialize_cpu_memory();
void initialize_gpu_memory();
void free_cpu_memory();
void free_gpu_memory();
void start_daemon();
void stop_daemon();
void start_client();
void sigterm_signal_handler(int signum);
void sigint_signal_handler(int signum);
time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes);

void daemon_pid_write_to_file(pid_t pid);
unsigned char daemon_pid_file_exists();
pid_t daemon_pid_read_from_file();
void daemon_pid_delete_file();


#endif
