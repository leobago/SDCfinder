#include <stdbool.h>
#include <assert.h>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "MemoryReliability_decl.cuh"
#endif
#include "MemoryReliability_decl.h"


unsigned char* random_ptr_inside(unsigned char* start, unsigned long long size)
{
    unsigned int ruibytes;
    int ribytes = rand();
    memcpy(&ruibytes, &ribytes, sizeof(ribytes)); // TODO check if a cast would do the same
    ruibytes = ruibytes % size;
    return start + ruibytes;
}

unsigned char* align_to_addrvalue(unsigned char* ptr_data)
{
    uintptr_t ptrmask = sizeof(uintptr_t)-1;// TODO should we take the sizeof ADDRVALUE here?
    return (unsigned char*) ((uintptr_t)ptr_data & ~ptrmask);
}

bool inject_dice_roll()
{
    int riter = rand()%10 + 1;
    return riter == 5;
}

void inject_cpu_error(unsigned char* start, ADDRVALUE expected_value, Strategy_t s)
{
#ifdef INJECT_ERR
    unsigned char *ptr_data = NULL;

    if (inject_dice_roll()) { // FIXME the caller should decide to inject or not
        switch (s) {
        case SIMPLE:
            break;
        case ZERO:
            ptr_data = random_ptr_inside(start, NumBytesCPU);
            memset(ptr_data, 0x1, 1);
            // TODO ask if this is correct;
            // We write to an address and print a different one.
            ptr_data = align_to_addrvalue(ptr_data);
            break;
        case RANDOM:
            ptr_data = align_to_addrvalue(random_ptr_inside(start, NumBytesCPU));
            *ptr_data = ~expected_value;
            break;
        default:
            assert(false);
        }

        if (ptr_data != NULL) {
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
    unsigned char *ptr_data = NULL;
    cudaError_t err;

    if (inject_dice_roll()) { // FIXME the caller should decide to inject or not
        switch (s) {
        case SIMPLE:
            break;
        case ZERO:
            ptr_data = random_ptr_inside(start, NumBytesGPU);
            err = cudaMemset(ptr_data, 0x1, 1);
            if (err != cudaSuccess)
            {
                char msg[255];
                sprintf(msg, "ERROR,Cannot memset ptr_data to 0x1 (%s:%s).",
                        cudaGetErrorName(err), cudaGetErrorString(err));
                log_message(msg);
            }
            // TODO ask if this is correct;
            // We write to an address and print a different one.
            ptr_data = align_to_addrvalue(ptr_data);
            break;
        case RANDOM:
            ptr_data = align_to_addrvalue(random_ptr_inside(start, NumBytesGPU));
            ADDRVALUE new_value = ~expected_value;
            // TODO ask if correct; the CPU version writes one byte because dereferences a char*
            err = cudaMemcpy(ptr_data, &new_value, 1, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                char msg[255];
                sprintf(msg, "ERROR,Cannot memcpy ptr_data to GPU (%s:%s).",
                        cudaGetErrorName(err), cudaGetErrorString(err));
                log_message(msg);
            }
            break;
        default:
            assert(false);
        }

        if (ptr_data != NULL) {
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
    switch (type) {
    case SIMPLE:
        simple_memory_test();
        break;
    case ZERO:
        zero_one_test();
        break;
    case RANDOM:
        random_pattern_test();
        break;
    default:
        assert(false);
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
    if (is_initialized(gpu_mem)) return;

    cudaError_t err;
    err = cudaMalloc(&gpu_mem, NumBytesGPU);
    if (err != cudaSuccess)
    {
        char msg[255];
        sprintf(msg, "ERROR_INFO,Cannot allocate %llu bytes of GPU memory (%s:%s).",
                NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
        log_message(msg);
    } else {
        err = cudaMemset(gpu_mem, 0x0, NumBytesGPU);
        if (err != cudaSuccess)
        {
            char msg[255];
            sprintf(msg, "ERROR_INFO,Cannot memset %llu bytes of GPU memory (%s:%s).",
                    NumBytesGPU, cudaGetErrorName(err), cudaGetErrorString(err));
            log_message(msg);
        }
    }

    if (err != cudaSuccess) {
        sleep(2);
        if (daemon_pid_file_exists())
        {
            daemon_pid_delete_file();
        }
        exit(EXIT_FAILURE);
    }
#endif
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




