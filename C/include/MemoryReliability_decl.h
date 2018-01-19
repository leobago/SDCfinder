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
 *  This file containis all include definition needed for the SDC tool.
 *  Further- more it contains the init values for all global variables.
 *  It also contains constant value definitions, global derived type
 *  definitions and function header.
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

/*! \brief Numerator to select test strategy
 *  
 *  SIMPLE  -> simple_memory_test()
 *  ZERO    -> zero_one_test()
 *  RANDOM  -> random_pattern_test()
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

/*! \brief Injects error in CPU memory
 *  \param start            [in]    pointer to memory allocation
 *  \param expected_value   [in]    expected bit pattern in memory
 *  \param s                [in]    monitoring strategy
 *
 *  \snippetlineno lib/injection.c inject_cpu_error
 **/
void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);

/*! \brief Injects error in GPU memory
 *  \param start            [in]    pointer to memory allocation
 *  \param expected_value   [in]    expected bit pattern in memory
 *  \param s                [in]    monitoring strategy
 *
 *  \snippetlineno lib/injection.c inject_gpu_error
 **/
void inject_gpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s);

/*! \brief Translates a virtual address to the physical address
 *  \param virt_addr        [in]    virtual address
 *  \return physical address mapped by virtual address
 *
 *  \snippetlineno lib/addresstranslation.c virtual_to_physical_address
 **/
uintptr_t virtual_to_physical_address(uintptr_t virt_addr);

/*! \brief Writes message in OutFile 
 *  \param message          [in]    message string
 *
 *  writes in the following order to the output file (set in OutFile) in
 *  one line: formatted time, time in ns, message, hostname, temperature 
 *
 *  \snippetlineno lib/logging.c log_message
 **/
void log_message(char* message);

/*! \brief Reports SDC event in OutFile 
 *  \param message          [in]    message string
 *
 *  writes in the following order to the output file (set in OutFile) in
 *  one line: formatted time, time in ns, hostname, virtual address,
 *  pattern found, pattern expected, temperature, physical address
 *
 *  notice: temperature and physical address will only be shown if
 *  application runs with administrator rights.
 *
 *  \snippetlineno lib/logging.c log_error
 **/
void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value);

/*! \brief Parser for commandline arguments 
 *  \param argc             [in]    argc from main
 *  \param argv             [in]    argv from main
 *
 *  \snippetlineno MemoryReliability.c parse_arguments
 **/
bool parse_arguments(int argc, char* argv[]);

/*! \brief Prints application help string
 *  \param argv             [in]    argv[0] from main
 *
 *  \snippetlineno MemoryReliability.c print_usage
 **/
void print_usage(char* program_name);

/*! \brief Checks for running daemon
 *  
 *  called during initialization if tool is supposed to run as daemon.
 *  If an instance of the daemon is already running, the client is
 *  exited and a warning is printed to stderr.
 *
 *  \snippetlineno lib/daemon.c check_no_daemon_running
 **/
void check_no_daemon_running();

/*! \brief Allocates CPU memory region and set to 0 
 *
 *  \snippetlineno lib/tools.c initialize_cpu_memory
 **/
void initialize_cpu_memory();

/*! \brief Allocates GPU memory region and set to 0 
 *
 *  \snippetlineno lib/tools.c initialize_gpu_memory
 **/
void initialize_gpu_memory();

/*! \brief De-allocates CPU memory. 
 *
 *  \snippetlineno lib/tools.c free_cpu_memory
 **/
void free_cpu_memory();

/*! \brief De-allocates GPU memory. 
 *
 *  \snippetlineno lib/tools.c free_gpu_memory
 **/
void free_gpu_memory();

/*! \brief Allocates GPU memory region and set to 0. 
 *
 *  \snippetlineno MemoryReliability.c start_daemon
 **/
void start_daemon();

/*! \brief Start client as daemon.
 *
 *  the main process creates a fork (daemon) and exits.  The fork
 *  installs the signal handler and starts the test loop.
 *
 *  \snippetlineno MemoryReliability.c stop_daemon
 **/
void stop_daemon();

/*! \brief Start daemon thread and exit.
 *
 *  installs the signal handler (SIGTERM and SIGINT) and starts the test
 *  loop.
 *
 *  \snippetlineno MemoryReliability.c start_client
 **/
void start_client();

/*! \brief Log signal type and exit. 
 *  \param signum signal type
 *
 *  \snippetlineno lib/daemon.c sigterm_signal_handler
 **/
void sigterm_signal_handler(int signum);

/*! \brief Log signal type and exit. 
 *
 *  \snippetlineno lib/daemon.c sigint_signal_handler
 **/
void sigint_signal_handler(int signum);

/*! \brief Sets formatted time string.
 *  \param  out_time_formatted  [out]   formatted time string
 *  \param  num_bytes           [in]    maximum length of time string
 *
 *  \snippetlineno lib/logging.c get_formatted_timestamp
 **/
time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes);

/*! \brief Start memory test loop.
 *  \param  type                [out]   strategy type
 *
 *  the strategy may be SIMPLE, ZERO or RANDOM.
 *
 *  \snippetlineno lib/memcheck.c memory_test_loop
 **/
void memory_test_loop(Strategy_t type);

/*! \brief Simple memory test
 *  
 *  the memory region is initialized to 0 at start. the region is
 *  devided into 32/64 bit words, depending on the architecture. On
 *  start, the memory is initialized to 0. After the first iteration
 *  each block in memory is set to 1, after the i'th iteration each
 *  block is set to i.
 *
 *  \snippetlineno lib/memcheck.c simple_memory_test
 **/
void simple_memory_test();

/*! \brief Zero one memory test
 *  
 *  the memory region is initialized to 0 at start. each iteration the
 *  bits in the region are flipped. Hence 0->1, 1->0, 0->1, etc.
 *
 *  \snippetlineno lib/memcheck.c zero_one_test
 **/
void zero_one_test();

/*! \brief Random pattern memory test
 *  
 *  the memory is initialized with an 8 bit pattern. The pattern is
 *  randomly generated at start of the client. each iteration, the bits
 *  in the memory region are flipped.
 *
 *  \snippetlineno lib/memcheck.c random_pattern_test
 **/
void random_pattern_test();

/*! \brief Scans memory for SDC (CPU)
 *  \param  buffer              [in]    pointer to allocation
 *  \param  num_bytes           [in]    size of allocation
 *  \param  expected_value      [in]    expected memory pattern
 *  \param  new_value           [in]    new pattern
 *
 *  This function scans the allocated memory region for SDC. Before the
 *  call, the memory region is partitioned to uintptr_t size words and
 *  set set to a defined pattern. Within the function, the allocated
 *  memory is compared against this pattern. If the expected value and
 *  the found value do not coincide, an error is logged as SDC. After
 *  the call the memory region will be set to the new pattern
 *  'new_value'
 *
 *  \snippetlineno lib/memcheck.c check_cpu_mem
 **/
bool check_cpu_mem(ADDRVALUE* buffer, unsigned long long num_bytes, 
        ADDRVALUE expected_value, ADDRVALUE new_value);

/*! \brief Scans memory for SDC (GPU) 
 *  \param  buffer              [in]    pointer to allocation
 *  \param  num_bytes           [in]    size of allocation
 *  \param  expected_value      [in]    expected memory pattern
 *  \param  new_value           [in]    new pattern
 *
 *  This function scans the allocated memory region for SDC. Before the
 *  call, the memory region is partitioned to uintptr_t size words and
 *  set set to a defined pattern. Within the function, the allocated
 *  memory is compared against this pattern. If the expected value and
 *  the found value do not coincide, an error is logged as SDC. After
 *  the call the memory region will be set to the new pattern
 *  'new_value'
 *
 *  \snippetlineno lib/memcheck.c check_gpu_mem
 **/
bool check_gpu_mem(ADDRVALUE* buffer, unsigned long long num_bytes,
        ADDRVALUE expected_value, ADDRVALUE new_value);

/*! \brief Store pid of daemon 
 *  \param  pid                 [in]    pid of damon thread
 *
 *  This function writes the pid of the daemon process into a text file
 *
 *  \snippetlineno lib/daemon.c daemon_pid_write_to_file
 **/
void daemon_pid_write_to_file(pid_t pid);

/*! \brief Checks if pid.txt exists 
 *
 *  \snippetlineno lib/daemon.c daemon_pid_file_exists
 **/
unsigned char daemon_pid_file_exists();

/*! \brief Reads daemon pid from file pid.txt. 
 *
 *  \snippetlineno lib/daemon.c daemon_pid_read_from_file
 **/
pid_t daemon_pid_read_from_file();

/*! \brief Removes file pid.txt. 
 *
 *  \snippetlineno lib/daemon.c daemon_pid_delete_file
 **/
void daemon_pid_delete_file();

#endif
