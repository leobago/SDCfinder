#include "MemoryReliability_decl.h"

/*
   +===========================================================================+
   |    DEFINITIONS                                                            |
   +===========================================================================+
   */

const unsigned int  TCC_ACT_TEMP        = 100;                      // TCC activation temperature
const unsigned int  TEMP_FIELD_LOW_BIT  = 16;                       // low bit of 6 bit temp value
const unsigned int  TEMP_FIELD_OFFSET   = 412;                      // offset in /dev/cpu/#cpu/msr file
uint64_t            TEMP_FIELD_MASK     = 0x00000000003f0000;       // selects 6 bit field[22:16]
const unsigned int  KILO                = 1024;                     // KB factor
const unsigned int  MEGA                = 1048576;                  // MB factor
const unsigned int  GIGA                = 1073741824;               // GB factor

/*
   +===========================================================================+
   |    STATIC VARIABLES                                                       |
   +===========================================================================+
   */

char                PidFileName[255]    = "pid.txt";                // stores the pid of daemon
char                OutFile[255]        = "MemoryReliability.log";  // Memory error log file
char                WarnFile[255]       = "MemoryReliability.warn"; // Logs if error exceeds warning rate 
char                ErrFile[255]        = "MemoryReliability.err";  // Logs runtime errors
char                DbgFile[255]        = "MemoryReliability.dbg";  // Logs debug information

unsigned int        WarnRate            = 0;                        // Threshold that triggers a warning
unsigned int        NumErrors           = 0;                        // Current number of errors

void*               Mem                 = NULL;                     // Keeps the allocated memory region to check

unsigned int        SleepTime           = 0;                        // Pause between checking for errors
bool                ExitNow             = false;                    // Triggers daemon to exit 
int                 IsDaemonStart       = 0;                        // Run as daemon if 1
int                 IsDaemonStop        = 0;                        // Kill Daemon if 1
