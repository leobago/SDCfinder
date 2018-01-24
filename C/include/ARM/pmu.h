/*
 * pmu.h
 *
 * This file declares the methods for accessing ARM
 * Performance Monitoring Unit (PMU) such as adding
 * counters to monitor, starting and stopping the monitoring.
 *
 * HOW TO USE
 * ----------
 *
 * uint64_t results[7];
 * addCounter(InstRetired);
 * addCounter(BranchMisspred);
 * addCounter(L1DRead);
 * addCounter(L1DWrite);
 * addCounter(LocalMemErrs);
 * openCounters();
 * startCounters();
 *
 * compute(...);
 *
 * uint32_t nres = stopCounters(results);
 * if (nres != 6){
 *     fprintf(stderr, "Something went totally wrong\n");
 *     return -1;
 * }
 *
 * printf("%s: %llu\n",counter_desc[Cycles].name,results[0]);
 * printf("%s: %llu\n",counter_desc[InstRetired].name,results[1]);
 * printf("%s: %llu\n",counter_desc[BranchMisspred].name,results[2]);
 * printf("%s: %llu\n",counter_desc[L1DRead].name,results[3]);
 * printf("%s: %llu\n",counter_desc[L1DWrite].name,results[4]);
 * printf("%s: %llu\n",counter_desc[LocalMemErrs].name,results[5]);
 *
 *  Created on: May 19, 2015
 *      Author: Ferad Zyulkyarov <ferad.zyulkyzrov@bsc.es>
 */

#ifndef __PMU_H_
#define __PMU_H_

#include <stdint.h>

/**
 * enum type to define labels for the PMU counters
 */
typedef enum counterindexes {
	Cycles = 0,
	InstRetired,
	DTLBRefill,
	BranchMisspred,
	MemAccess,
	L2Access,
	L2Miss,
	BusAccess,
	InstSpec,
	L1DRead,
	L1DWrite,
	L1DReadMisses,
	L1DWriteMisses,
	L2DRead,
	L2DWrite,
	L2DReadMisses,
	L2DWriteMisses,
	LDST,
	Int,
	VFP,
	Branches,
	LocalMemErrs,
	Last_CntIdx,
} CntIdx;

/**
 * Struct used to describe each counter with id, type and name.
 */
typedef struct counter {
	uint32_t id;
	uint32_t type;
	const char* name;
} counter_t;

/**
 * One description record for each counter in CntIdx.
 */
extern counter_t counter_desc[];

/**
 * Add a counter to the set of counters to be monitored.
 * The first counter is hardwired to cycles, we have 6 extra
 * slots left.
 *
 * @param newCntr the counter to be monitored.
 *
 * @return 0 if this operation failed or 1 if this operation succeeded.
 */
int addCounter(CntIdx newCntr);

/**
 * Open the set of counter to be monitored.
 *
 * @return 0 if this operation failed or 1 if this operation succeeded.
 */
int openCounters();

/**
 * Start the counters.
 *
 * @return 0 if this operation failed or 1 if this operation succeeded.
 */
int startCounters();

/**
 * Read the counters.
 *
 * @param counterValues an array large enough to hold the values
 * of all counters including the implicit cycle counter.
 *
 * @return the number of counters which are read.
 */
uint32_t readCounters(uint64_t counterValues[]);

/**
 * Stop monitoring the performance counters and read the counters.
 *
 * @param counterValues an array large enough to hold the values
 * of all counters including the implicit cycle counter.
 *
 * @return the number of counters which are read.
 */
uint32_t stopCounters(uint64_t counterValues[]);

/**
 * Initializes and starts only the LocalMemErrors counter.
 *
 * @return 1 if successful and 0 if not successful.
 */
int startLocalMemErrorsCounters();

/**
 * Stops the LocalMemErrors counters and reads the values.
 *
 * @return the value of the LocalMemErrs counter. If unsuccessful
 * returns -1.
 */
int stopLocalMemErrorsCounters();

/**
 * Reads the LocalMemErrors counter.
 *
 * @return the value of the LocalMemErrs counter. If unsuccessful
 * returns -1.
 */
int readLocalMemErrorsCounters();

#endif /* __PMU_H_ */

