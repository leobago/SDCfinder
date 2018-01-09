#ifndef _MEMORYRELIABILITY_DECL_H
#define _MEMORYRELIABILITY_DECL_H

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
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

#include "addresstranslation.h"
#include "pmu.h"
#include "types.h"
#include "logging.h"

// DEFINITIONS

extern const unsigned int KILO;//                 = 1024;
extern const unsigned int MEGA;//                 = 1048576;
extern const unsigned int GIGA;//                 = 1073741824;

// STATIC VARIABLES

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

void memory_test_loop(Strategy_t type);

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
