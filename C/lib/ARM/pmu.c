/*
 * pmu.c
 *
 * This file defines the methods for accessing ARM
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
 * compute(...); // the method to be monitored
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

#include "../include/pmu.h"
#include <stdlib.h>
#include <linux/perf_event.h>
#include <syscall.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

counter_t counter_desc[Last_CntIdx] = {
	{PERF_COUNT_HW_CPU_CYCLES, PERF_TYPE_HARDWARE, "Cycles" },
	{PERF_COUNT_HW_INSTRUCTIONS, PERF_TYPE_HARDWARE, "InstRetired" },
	{/* L1D_TLB_REFILL */      0x03, PERF_TYPE_RAW, "DTLBRefill" },
	{/* BR_MIS_PRED */         0x10, PERF_TYPE_RAW, "BranchMisspred" },
	{/* MEM_ACCESS */          0x13, PERF_TYPE_RAW, "MemAccess" },
	{/* L2D_CACHE */           0x16, PERF_TYPE_RAW, "L2DAccess" },
	{/* L2D_CACHE_REFILL */    0x17, PERF_TYPE_RAW, "L2DMiss" },
	{/* BUS_ACCESS */          0x19, PERF_TYPE_RAW, "BusAccess" },
	{/* INST_SPEC */           0x1b, PERF_TYPE_RAW, "InstSpec" },
	{/* L1D_CACHE_LD */        0x40, PERF_TYPE_RAW, "L1DRead" },
	{/* L1D_CACHE_ST */        0x41, PERF_TYPE_RAW, "L1DWrite" },
	{/* L1D_CACHE_REFILL_LD */ 0x42, PERF_TYPE_RAW, "L1DReadMisses" },
	{/* L1D_CACHE_REFILL_ST */ 0x43, PERF_TYPE_RAW, "L1DWriteMisses" },
	{/* L2D_ACCESS_LD */       0x50, PERF_TYPE_RAW, "L2DRead" },
	{/* L2D_ACCESS_ST */       0x51, PERF_TYPE_RAW, "L2DWrite" },
	{/* L2D_CACHE_REFILL_LD */ 0x52, PERF_TYPE_RAW, "L2DReadMisses" },
	{/* L2D_CACHE_REFILL_ST */ 0x53, PERF_TYPE_RAW, "L2DWriteMisses" },
	{/* LDST_SPEC */           0x72, PERF_TYPE_RAW, "LDST" },
	{/* DP_SPEC */             0x73, PERF_TYPE_RAW, "Int" },
	{/* VFP_SPEC */            0x75, PERF_TYPE_RAW, "VFP" },
	{/* BR_IMMED_SPEC */       0x78, PERF_TYPE_RAW, "Branches" },
	{/* LOCAL_MEM_ERR */       0x1A, PERF_TYPE_RAW, "LocalMemErrors" },
};

static int counter_fds[7];
static CntIdx counters[7];
static int nCounters = 1;

int addCounter(CntIdx newCounter) {
	if (nCounters >= 7) {
		/* At most 6+1 counters are allowed */
		return 0;
	}

	/* We should check for duplicates */
	counters[nCounters++] = newCounter;

	return 1;
}

static inline long
perf_event_open(
	struct perf_event_attr *hw_event,
	pid_t pid,
	int cpu,
	int group_fd,
	unsigned long flags)
{
	int ret;

	ret = syscall(
			__NR_perf_event_open,
			hw_event,
			pid,
			cpu,
			group_fd, flags);

	return ret;
}

int sys_open_counters_n(CntIdx perfEvents[], int fds[], uint32_t n)
{
	struct perf_event_attr pe;

	uint32_t i;

	for(i = 0; i < n; i ++){
		memset(&pe, 0, sizeof(struct perf_event_attr));
		pe.type = counter_desc[perfEvents[i]].type;
		pe.config = counter_desc[perfEvents[i]].id;
		pe.size = sizeof(struct perf_event_attr);
		pe.disabled = 1;
		/* We do not want to account for kernel code */
		pe.exclude_kernel = 1;
		/* W do not want to account for hyper-visor code either */
		pe.exclude_hv = 1;

		fds[i] = perf_event_open(&pe, 0, -1, -1, 0);
		if (fds[i] == -1) {
			/* Error opening leader pe.config (%s), pe.config, counter_desc[perfEvents[i]].name */
			return 0;
		}
	}

	return 1;
}

int openCounters()
{
	return sys_open_counters_n(counters, counter_fds, nCounters);
}

int start_counters_n(int fd[], uint32_t n)
{
	uint32_t i;
	for(i = 0; i < n; i++){
		ioctl(fd[i], PERF_EVENT_IOC_RESET, 0);
		ioctl(fd[i], PERF_EVENT_IOC_ENABLE, 0);
	}

	return 1;
}

int startCounters()
{
	return start_counters_n(counter_fds, nCounters);
}

int read_counters_n(int fd[], uint64_t v[],uint32_t n)
{
	uint32_t i;
	uint32_t warn = 0;
	for(i = 0; i < n; i++){
		long long r;
		warn = (read(fd[i], &r, sizeof(long long)) != sizeof(long long)) || warn;
		v[i] = r;
	}
	if(warn)
	{
		/* Unexpected size while reading performance counters */
		return 2;
	}
	return 1;
}

uint32_t readCounters(uint64_t v[])
{
	read_counters_n(counter_fds,v,nCounters);
	return nCounters;
}

uint32_t stop_counters_n(int fd[], uint64_t v[],uint32_t n)
{
	uint32_t i;
	for(i = 0; i < n; i++)
	{
		ioctl(fd[i], PERF_EVENT_IOC_DISABLE, 0);
	}
	return readCounters(v);
}

uint32_t stopCounters(uint64_t v[])
{
	return stop_counters_n(counter_fds, v, nCounters);
}

int startLocalMemErrorsCounters()
{
	int status = 0;

	status = addCounter(LocalMemErrs);
	if (!status)
	{
		return status;
	}

	status = openCounters();
	if (!status)
	{
		return status;
	}

	status = startCounters();

	return status;
}

int stopLocalMemErrorsCounters()
{
	uint64_t counterValues[2];
	counterValues[0] = 0;
	counterValues[1] = 0;

	uint32_t nres = stopCounters(counterValues);
	if (nres != nCounters)
	{
		return -1;
	}

	return (int)counterValues[1];
}

int readLocalMemErrorsCounters()
{
	uint64_t counterValues[2];
	counterValues[0] = 0;
	counterValues[1] = 0;

	uint32_t nres = readCounters(counterValues);
	if (nres != nCounters)
	{
		return -1;
	}

	return (int)counterValues[1];
}
