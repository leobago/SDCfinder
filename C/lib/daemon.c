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

#include "MemoryReliability_decl.h"

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

void check_no_daemon_running()
{
    pid_t pid = daemon_pid_read_from_file();
    if (pid != 0)
    {
        fprintf(stderr, "It appears that an instance of this daemon is running because the %s pid file exists. If you are sure that there is no daemon running delete the pid file: %s.\n", PidFileName, PidFileName);
        exit(EXIT_FAILURE);
    }
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

