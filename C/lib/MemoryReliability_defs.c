#include "MemoryReliability_decl.h"

// DEFINITIONS
const unsigned int TCC_ACT_TEMP         = 100; // TCC activation temperature
const unsigned int TEMP_FIELD_LOW_BIT   = 16; // low bit of 6 bit temp value
const unsigned int TEMP_FIELD_OFFSET    = 412; // offset in /dev/cpu/#cpu/msr file
uint64_t TEMP_FIELD_MASK                = 0x00000000003f0000; // selects 6 bit field[22:16]
const unsigned int KILO                 = 1024;
const unsigned int MEGA                 = 1048576;
const unsigned int GIGA                 = 1073741824;

// STATIC VARIABLES
char PidFileName[255] = "pid.txt";
char TemperatureFileName[] = "/sys/class/thermal/thermal_zone0/temp";
char OutFile[255] = "MemoryReliability.log";
char WarningFile[255] = "MemoryReliability.dbg";
char ErrFile[255] = "MemoryReliability.err";

unsigned int WarningRate = 0;
unsigned int NumErrors = 0;

unsigned long long NumBytes = 0;
void* Mem = NULL;

unsigned int SleepTime = 0;
bool ExitNow = false;
int IsDaemonStart = 0;
int IsDaemonStop = 0;
