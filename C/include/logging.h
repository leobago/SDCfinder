#ifndef LOGGING_H
#define LOGGING_H

#include "types.h"

extern const unsigned int TCC_ACT_TEMP;//         = 100; // TCC activation temperature
extern const unsigned int TEMP_FIELD_LOW_BIT;//   = 16; // low bit of 6 bit temp value
extern const unsigned int TEMP_FIELD_OFFSET;//    = 412; // offset in /dev/cpu/#cpu/msr file
extern uint64_t TEMP_FIELD_MASK;//                = 0x00000000003f0000; // selects 6 bit field[22:16]


extern char PidFileName[255];// = "pid.txt";
extern char TemperatureFileName[255];// = "/sys/class/thermal/thermal_zone0/temp";
extern char OutFile[255];// = "MemoryReliability.log";
extern char WarningFile[255];// = "";
extern char ErrFile[255];// = "MemoryReliability.err";

extern unsigned int WarningRate;// = 0;
extern unsigned int NumErrors;// = 0;

char HostName[255];

void log_error(void* address, ADDRVALUE actual_value, ADDRVALUE expected_value);
void log_message(char* message);
void warn_for_errors();
void log_local_mem_errors(int local_mem_errors); // Logs local memory errors

#endif //LOGGING_H
