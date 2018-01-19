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

/** \file   MemoryReliability_decl.h
 *  \author Ferad Zyulkyarov
 *  \author Kai Keller
 *  \author Pau Farré
 *  \author Leonardo Bautista-Gomez
 *  \brief  Main header file and interface for MemoryReliability.c
 *
 *  This file containis all include definition needed for the SDC tool. Further-
 *  more it contains the init values for all global variables. It also contains
 *  constant value definitions, global derived type definitions and function header.
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

#if 0 // only for ARM
#include "pmu.h"
#endif

/*
   +===========================================================================+
   |    DEFINITIONS                                                            |
   +===========================================================================+
   */

extern const unsigned int KILO;             /*! pre-factor for kilo bytes                           */
extern const unsigned int MEGA;             /*! pre-factor for mega bytes                           */
extern const unsigned int GIGA;             /*! pre-factor for giga bytes                           */

/*
   +===========================================================================+
   |    GLOBAL VARIABLES                                                       |
   +===========================================================================+
   */

extern char PidFileName[255];               /*! filename to store prozess identifier (PID)          */
extern char OutFile[255];                   /*! filename of SDC event logging                       */
extern char WarnFile[255];                  /*! filename for warnings if SDC event amount is high   */
extern char ErrFile[255];                   /*! filename for runtime error logging                  */
extern char DbgFile[255];                   /*! filename to store debug information                 */

extern unsigned int WarnRate;               /*! threshold for log a warning                         */
extern unsigned int NumErrors;              /*! amount of SDC events detected                       */

extern const unsigned int MAX_ITER;         /*! security measure to prevent infinite loop execution */

extern unsigned int SleepTime;              /*! time between two check loops                        */
extern bool ExitNow;                        /*! flag for daemon to stop execution                   */
extern int IsDaemonStart;                   /*! flag if daemon should be started                    */
extern int IsDaemonStop;                    /*! flag if daemon should be stopped                    */

extern char HostName[255];                  /*! keeps the hostname of executing node                */   

extern unsigned long long NumBytesCPU;      /*! size of memory region to monitor for SDC in CPU     */
extern unsigned long long NumBytesGPU;      /*! size of memory region to monitor for SDC in GPU     */     
extern void* cpu_mem;                       /*! pointer to allocated CPU memory region              */
extern void* gpu_mem;                       /*! pointer to allocated GPU memory region              */

extern unsigned char mem_pattern;           /*! random generated pattern to initialize memory       */
extern bool CheckCPU;                       /*! flag if CPU memory is to monitor                    */
extern bool CheckGPU;                       /*! flag if GPU memory is to monitor                    */

/*
   +===========================================================================+
   |    TYPE DEFINITIONS                                                       |
   +===========================================================================+
   */

typedef uintptr_t ADDRVALUE;                /*! defines ADDRVALUE type                              */
typedef void* ADDRESS;                      /*! defines ADDRESS type                                */

/*! \brief numerator to select test strategy
 *  
 *  SIMPLE  -> simple_memory_test()
 *  ZERO    -> zero_one_test()
 *  RANDOM  -> random_pattern_test() *
 */
typedef enum Strategy {
    SIMPLE = 0,
    ZERO = 1,
    RANDOM = 2
} Strategy_t;

/*
   +===========================================================================+
   |    FUNCTION DECLARATIONS                                                  |
   +===========================================================================+
   */

/*! \brief injects error in CPU memory
 *  \param start            [in]    pointer to memory allocation
 *  \param expected_value   [in]    expected bit pattern in memory
 *  \param s                [in]    monitoring strategy
 **/
void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);

/*! \brief injects error in GPU memory
 *  \param start            [in]    pointer to memory allocation
 *  \param expected_value   [in]    expected bit pattern in memory
 *  \param s                [in]    monitoring strategy
 **/
void inject_gpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);

/*! \brief translates a virtual address to the physical address
 *  \param virt_addr        [in]    virtual address
 *  \return physical address mapped by virtual address
 **/
uintptr_t virtual_to_physical_address(uintptr_t virt_addr);

/*! \brief writes message in OutFile 
 *  \param message          [in]    message string
 *
 *  writes in the following order to the output file (set in OutFile) in one line:
 *  formatted time, time in ns, message, hostname, temperature 
 **/
void log_message(char* message);

/*! \brief reports SDC event in OutFile 
 *  \param message          [in]    message string
 *
 *  writes in the following order to the output file (set in OutFile) in one line:
 *  formatted time, time in ns, hostname, virtual address, pattern found, 
 *  pattern expected, temperature, physical address
 *
 *  notice: temperature and physical address will only be shown if application
 *  runs with administrator rights.
 **/
void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value);

/*! \brief parser for commandline arguments 
 *  \param argc             [in]    argc from main
 *  \param argv             [in]    argv from main
 **/
bool parse_arguments(int argc, char* argv[]);

/*! \brief prints application help string
 *  \param argv             [in]    argv[0] from main
 **/
void print_usage(char* program_name);

/*! \brief checks for running daemon
 *  
 *  called during initialization if tool is supposed to run as daemon.
 *  If an instance of the daemon is already running, the client is exited
 *  and a warning is printed to stderr.
 **/
void check_no_daemon_running();

/*! \brief allocates CPU memory region and set to 0 */
void initialize_cpu_memory();

/*! \brief allocates GPU memory region and set to 0 */
void initialize_gpu_memory();

/*! \brief de-allocates CPU memory. */
void free_cpu_memory();

/*! \brief de-allocates GPU memory. */
void free_gpu_memory();

/*! \brief allocates GPU memory region and set to 0. */
void start_daemon();

/*! \brief start client as daemon.
 *
 *  the main process creates a fork (daemon) and exits.
 *  The fork installs the signal handler and starts the test loop.
 **/
void stop_daemon();

/*! \brief start daemon thread and exit.
 *
 *  installs the signal handler (SIGTERM and SIGINT) and starts the test loop.
 **/
void start_client();

/*! \brief log signal type and exit. 
 *  \param signum signal type
 **/
void sigterm_signal_handler(int signum);

/*! \brief log signal type and exit. */
void sigint_signal_handler(int signum);

/*! \brief sets formatted time string.
 *  \param  out_time_formatted  [out]   formatted time string
 *  \param  num_bytes           [in]    maximum length of time string
 **/
time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes);

/*! \brief start memory test loop.
 *  \param  type                [out]   strategy type
 *
 *  the strategy may be SIMPLE, ZERO or RANDOM.
 **/
void memory_test_loop(Strategy_t type);

/*! \brief simple memory test
 *  
 *  the memory region is initialized to 0 at start. the region is devided into
 *  32/64 bit words, depending on the architecture. On start, the memory is 
 *  initialized to 0. After the first iteration each block in memory is set to
 *  1, after the i'th iteration each block is set to i.
 **/
void simple_memory_test();

/*! \brief zero one memory test
 *  
 *  the memory region is initialized to 0 at start. each iteration the bits
 *  in the region are flipped. Hence 0->1, 1->0, 0->1, etc.
 **/
void zero_one_test();

/*! \brief random pattern memory test
 *  
 *  the memory is initialized with an 8 bit pattern. The pattern is randomly
 *  generated at start of the client. each iteration, the bits in the memory region 
 *  are flipped.
 **/
void random_pattern_test();

/*! \brief scans memory for SDC 
 *  \param  buffer              [in]    pointer to allocation
 *  \param num_bytes            [in]    size of allocation
 *  \param                      [in]    expected memory pattern
 *  \param                      
 **/
bool check_cpu_mem(ADDRVALUE* buffer, unsigned long long num_bytes, 
        ADDRVALUE expected_value, ADDRVALUE new_value);

bool check_gpu_mem(ADDRVALUE* buffer, unsigned long long num_bytes,
        ADDRVALUE expected_value, ADDRVALUE new_value);

void daemon_pid_write_to_file(pid_t pid);

unsigned char daemon_pid_file_exists();

pid_t daemon_pid_read_from_file();

void daemon_pid_delete_file();

#endif
