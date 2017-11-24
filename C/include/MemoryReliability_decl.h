#ifndef _MEMORYRELIABILITY_DECL_H
#define _MEMORYRELIABILITY_DECL_H

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <errno.h>

#include "addresstranslation.h"
#include "pmu.h"

// DEFINITIONS
typedef uintptr_t ADDRVALUE;
typedef void* ADDRESS;

const unsigned int TCC_ACT_TEMP;//         = 100; // TCC activation temperature
const unsigned int TEMP_FIELD_LOW_BIT;//   = 16; // low bit of 6 bit temp value
const unsigned int TEMP_FIELD_OFFSET;//    = 412; // offset in /dev/cpu/#cpu/msr file
uint64_t TEMP_FIELD_MASK;//                = 0x00000000003f0000; // selects 6 bit field[22:16]
const unsigned int KILO;//                 = 1024;
const unsigned int MEGA;//                 = 1048576;
const unsigned int GIGA;//                 = 1073741824;

// STATIC VARIABLES
char PidFileName[255];// = "pid.txt";
char TemperatureFileName[255];// = "/sys/class/thermal/thermal_zone0/temp";
char OutFile[255];// = "MemoryReliability.log";
char WarningFile[255];// = "";
char ErrFile[255];// = "MemoryReliability.err";

unsigned int WarningRate;// = 0;
unsigned int NumErrors;// = 0;

char HostName[255];

unsigned long long NumBytes;// = 0;
void* Mem;// = NULL;
unsigned char mem_pattern;

unsigned int SleepTime;// = 0;
unsigned char ExitNow;// = 0;
unsigned char IsDaemonStart;// = 0;
unsigned char IsDaemonStop;// = 0;

// FUNCTIONS
void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value);
void log_message(char* message);
void warn_for_errors();
void log_local_mem_errors(int local_mem_errors); // Logs local memory errors

void simple_memory_test(void* buffer, unsigned int num_bytes);
void zero_one_test(void* buffer, unsigned int num_bytes);
void random_pattern_test(void* buffer, unsigned int num_bytes);

unsigned char parse_arguments(int argc, char* argv[]);
void print_usage();
void check_no_daemon_running();
void initialize_memory();
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

int read_temperature();

#endif
