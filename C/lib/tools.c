#include <stdbool.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "MemoryReliability_decl.cuh"
#endif
#include "MemoryReliability_decl.h"

void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s)
{
#ifdef INJECT_ERR
    if (s != SIMPLE)
    {
        int riter = rand()%10 + 1;
        if (riter == 5) {
            unsigned int ruibytes;
            int ribytes = rand();
            memcpy(&ruibytes, &ribytes, sizeof(ribytes));
            ruibytes = (ruibytes%NumBytesCPU);
            unsigned char *ptr_data = start + ruibytes;
            if (s == ZERO) {
                memset(ptr_data, 0x1, 1);
            }
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask);
            if (s == RANDOM) {
                *ptr_data = ~expected_value;
            }
            char msg[1000];
            snprintf(msg,1000,"Inject error at %p (CPU)", ptr_data);
            log_message(msg);
        }
    }
#endif
}

void inject_gpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s)
{
#if defined(INJECT_ERR) && defined(USE_CUDA)
    if (s != SIMPLE)
    {
        int riter = rand()%10 + 1;
        if (riter == 5) {
            unsigned int ruibytes;
            int ribytes = rand();
            memcpy(&ruibytes, &ribytes, sizeof(ribytes));
            ruibytes = (ruibytes%NumBytesGPU);
            unsigned char* ptr_data = start + ruibytes;
            if (s == ZERO) {
                cudaError_t err = cudaMemset(ptr_data, 0x1, 1);
                if (err != cudaSuccess)
                {
                    char msg[255];
                    sprintf(msg, "ERROR,Cannot memset ptr_data to 0x1 (%s:%s).",
                            cudaGetErrorName(err), cudaGetErrorString(err));
                    log_message(msg);
                }
            }
            uintptr_t ptrmask = sizeof(uintptr_t)-1;
            ptr_data = (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask);
            ADDRVALUE new_value = ~expected_value;
            if (s == RANDOM) {
                cudaError_t err = cudaMemcpy(ptr_data, &new_value, sizeof(ADDRVALUE), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    char msg[255];
                    sprintf(msg, "ERROR,Cannot memcpy ptr_data to GPU (%s:%s).",
                            cudaGetErrorName(err), cudaGetErrorString(err));
                    log_message(msg);
                }
            }
            char msg[1000];
            snprintf(msg,1000,"Inject error at %p (GPU)", ptr_data);
            log_message(msg);
        }
    }
#endif
}

bool check_cpu_mem(ADDRVALUE* buffer,
                   unsigned long long num_bytes,
                   ADDRVALUE expected_value,
                   ADDRVALUE new_value)
{
    const long unsigned num_words = num_bytes / sizeof(ADDRVALUE);
    for (long unsigned word = 0; word < num_words; word++)
    {
        ADDRVALUE actual_value = buffer[word];
        if (actual_value != expected_value)
        {
            log_error(&buffer[word], actual_value, expected_value);
        }

        buffer[word] = new_value;
    }
    return true;
}

bool check_gpu_mem(ADDRVALUE* buffer,
                   unsigned long long num_bytes,
                   ADDRVALUE expected_value,
                   ADDRVALUE new_value)
{
#if defined(USE_CUDA)
    check_gpu_stub(buffer, num_bytes, expected_value, new_value);
#endif
    return true;
}

void simple_memory_test()
{
	//
	// Start the HW counters for the LocalMemErrors
	//
	int local_mem_errors = startLocalMemErrorsCounters();
	if (!local_mem_errors)
	{
		char msg[] = "Cannot start LocalMemError counters.";
		log_message(msg);
		//ExitNow = 1;
		if (daemon_pid_file_exists())
		{
			daemon_pid_delete_file();
		}
	}

    ADDRVALUE expected_value = 0;

	//
	// The main loop
	//
	while(ExitNow == 0)
	{
        check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, expected_value+1);
        check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, expected_value+1);

        expected_value++;

		local_mem_errors = readLocalMemErrorsCounters();
		log_local_mem_errors(local_mem_errors);

		if (SleepTime > 0 && ExitNow == 0)
		{
			sleep(SleepTime);
		}
	}

	local_mem_errors = stopLocalMemErrorsCounters();
	log_local_mem_errors(local_mem_errors);
}

void zero_one_test()
{
    //
    // Start the HW counters for the LocalMemErrors
    //
    int local_mem_errors = startLocalMemErrorsCounters();
    if (!local_mem_errors)
    {
        char msg[] = "Cannot start LocalMemError counters.";
        log_message(msg);
        //ExitNow = 1;
        if (daemon_pid_file_exists())
        {
            daemon_pid_delete_file();
        }
    }

    ADDRVALUE expected_value = 0x0;
    //
    // The main loop
    //
    while(ExitNow == 0)
    {
        //
        // Simulate error at random position and random iteration.
        //
        inject_cpu_error(cpu_mem, expected_value, ZERO);
        inject_gpu_error(gpu_mem, expected_value, ZERO);

        check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, ~expected_value);
        check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, ~expected_value);

        expected_value = ~expected_value;

        local_mem_errors = readLocalMemErrorsCounters();
        log_local_mem_errors(local_mem_errors);

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }
    }

    local_mem_errors = stopLocalMemErrorsCounters();
    log_local_mem_errors(local_mem_errors);
}

void random_pattern_test()
{
    //
    // Start the HW counters for the LocalMemErrors
    //
    int local_mem_errors = startLocalMemErrorsCounters();
    if (!local_mem_errors)
    {
        char msg[] = "Cannot start LocalMemError counters.";
        log_message(msg);
        //ExitNow = 1;
        //if (daemon_pid_file_exists())
        //{
        //    daemon_pid_delete_file();
        //}
    }

    ADDRVALUE expected_value;
    memset(&expected_value, mem_pattern, sizeof(uint64_t));
    if (CheckCPU)
    {
        memset(cpu_mem, mem_pattern, NumBytesCPU);
    }
#if defined(INJECT_ERR) && defined(USE_CUDA)
    if (CheckGPU)
    {
        cudaError_t err = cudaMemset(gpu_mem, mem_pattern, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR,Cannot memset ptr_data to 0x1 (%s:%s).",
                         cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);
        }
    }
#endif

    //
    // The main loop
    //
    while (ExitNow == 0)
    {
        // 
        // Simulate error at random position and random iteration.
        //
        if (CheckCPU)
            inject_cpu_error(cpu_mem, expected_value, RANDOM);
        if (CheckGPU)
            inject_gpu_error(gpu_mem, expected_value, RANDOM);

        if (CheckCPU)
            check_cpu_mem(cpu_mem, NumBytesCPU, expected_value, ~expected_value);
        if (CheckGPU)
            check_gpu_mem(gpu_mem, NumBytesGPU, expected_value, ~expected_value);

        expected_value = ~expected_value;

        local_mem_errors = readLocalMemErrorsCounters();
        log_local_mem_errors(local_mem_errors);

        if (SleepTime > 0 && ExitNow == 0)
        {
            sleep(SleepTime);
        }
    }

    local_mem_errors = stopLocalMemErrorsCounters();
    log_local_mem_errors(local_mem_errors);
}

void memory_test_loop(Strategy_t type)
{
    if (type == SIMPLE) {
        simple_memory_test();
    } else if (type == ZERO) {
        zero_one_test();
    } else if (type == RANDOM) {
        random_pattern_test();
    }
}

void check_no_daemon_running()
{
	pid_t pid = daemon_pid_read_from_file();
	if (pid != 0)
	{
		fprintf(stderr, "It appears that an instance of this daemon is running because the %s pid file exists. If you are sure that there is no daemon running delete the pid file: %s.\n", PidFileName, PidFileName);
		exit(EXIT_FAILURE);
	}
}

bool is_initialized(void* ptr)
{
    return ptr != NULL;
}

void initialize_cpu_memory()
{
    if (!is_initialized(cpu_mem))
    {
        cpu_mem = malloc(NumBytesCPU);
        if (cpu_mem == NULL)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot allocate %llu number of CPU memory.", NumBytesCPU);
            log_message(msg);
            sleep(2);
            if (daemon_pid_file_exists())
            {
                daemon_pid_delete_file();
            }
            exit(EXIT_FAILURE);
        }
        memset(cpu_mem, 0x0, NumBytesCPU);
    }
}

void initialize_gpu_memory()
{
#ifdef USE_CUDA
    if (!is_initialized(gpu_mem))
    {
        cudaError_t err;
        err = cudaMalloc(&gpu_mem, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot allocate %llu bytes of GPU memory (%s:%s).",
                    NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);

            goto fn_fatal_error;
        }
        err = cudaMemset(gpu_mem, 0x0, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot memset %llu bytes of GPU memory (%s:%s).",
                    NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);

            goto fn_fatal_error;
        }
    }
#endif
    return;

  fn_fatal_error:
    sleep(2);
    if (daemon_pid_file_exists())
    {
        daemon_pid_delete_file();
    }
    exit(EXIT_FAILURE);
}

void free_cpu_memory()
{
    free(cpu_mem);
}

void free_gpu_memory()
{
#ifdef USE_CUDA
    cudaError_t err;
    err = cudaFree(gpu_mem);
    if (err != cudaSuccess)
    {
        char msg[255];
        sprintf(msg, "ERROR_INFO,Cannot free GPU memory (%s:%s).",
                    cudaGetErrorName(err), cudaGetErrorString(err));
        log_message(msg);

        sleep(2);
        if (daemon_pid_file_exists())
        {
            daemon_pid_delete_file();
        }
        exit(EXIT_FAILURE);
    }
#endif
}

void sigterm_signal_handler(int signum)
{
	ExitNow = 1;
	log_message("STOP,SIGTERM");
}

void sigint_signal_handler(int signum)
{
	ExitNow = 1;
	log_message("STOP,SIGINT");
}

time_t get_formatted_timestamp(char* out_time_formatted, unsigned char num_bytes)
{
	time_t rawtime;
	struct tm *info;
	time(&rawtime);
	info = localtime(&rawtime);
	strftime(out_time_formatted, num_bytes, "%x - %H:%M:%S", info);
	return rawtime;
}

void daemon_pid_write_to_file(pid_t pid)
{
	FILE *f = fopen(PidFileName, "w+");
	if (!f)
	{
		fprintf(stderr, "ERROR: Cannot create a daemon pid file.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(f, "%d", pid);
	fclose(f);
}

unsigned char daemon_pid_file_exists()
{
	unsigned char is_file_exists = 0;

	FILE *f = fopen(PidFileName, "w");
	if (f)
	{
		is_file_exists = 1;
		fclose(f);
	}

	return is_file_exists;
}

pid_t daemon_pid_read_from_file()
{
	pid_t pid = 0;
	int fscanStatus = 0;
	FILE *f = fopen(PidFileName, "r");
	if (f)
	{
		int pid_int = 0;
		fscanStatus = fscanf(f, "%d", &pid_int);
		if (fscanStatus <= 0)
		{
		    fprintf(stderr, "ERROR: Cannot fscanf file %s for pid.\n", PidFileName);
		}
		pid = (pid_t)pid_int;
		fclose(f);
	}
	return pid;

}
void daemon_pid_delete_file()
{
	int delete_status = remove(PidFileName);
	if (delete_status != 0)
	{
		fprintf(stderr, "ERROR: Cannot delete the file %s. Probably no such file exists.\n", PidFileName);
	}
}

int read_temperature()
{
    int fd, fd_err, core = 0, ierr;
    uint64_t temp;
    char msr_file[32];
    memset(msr_file, 0, 32);

#ifdef SYS_getcpu
    if (syscall(SYS_getcpu, &core, NULL, NULL) < 0) {
        core = 0;
    }
#endif
    
    fd_err = open(ErrFile, O_WRONLY|O_CREAT|O_APPEND, 0666);
    
    snprintf(msr_file, 31, "/dev/cpu/%u/msr", core);
    fd = open(msr_file, O_RDONLY);
    if (fd == -1) {
        ierr = errno;
        dprintf(fd_err, "could not OPEN msr file: %s\n", strerror(ierr));
        ierr = 0;
        return -1;
    }
    
    ierr = pread(fd, &temp, sizeof(temp), TEMP_FIELD_OFFSET);
    if (ierr == -1) {
        ierr = errno;
        dprintf(fd_err, "could not READ msr file: %s\n", strerror(ierr));
        ierr = 0;
        return -1;
    }

    temp &= TEMP_FIELD_MASK;
    temp >>= TEMP_FIELD_LOW_BIT;
    temp = TCC_ACT_TEMP - temp;

    close(fd_err);
    close(fd);

	return temp;
}


