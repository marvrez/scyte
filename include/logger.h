#ifndef LOGGER_H
#define LOGGER_H

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <execinfo.h>
#include <sys/timeb.h>

#define RESET   ((const char*)"\033[0m")
#define BOLDBLACK   ((const char*)"\033[1m\033[30m")      /* Bold Black */
#define BOLDRED     ((const char*)"\033[1m\033[31m")      /* Bold Red */
#define BOLDGREEN   ((const char*)"\033[1m\033[32m")      /* Bold Green */
#define BOLDYELLOW  ((const char*)"\033[1m\033[33m")      /* Bold Yellow */
#define BOLDBLUE    ((const char*)"\033[1m\033[34m")      /* Bold Blue */
#define BOLDMAGENTA ((const char*)"\033[1m\033[35m")      /* Bold Magenta */
#define BOLDCYAN    ((const char*)"\033[1m\033[36m")      /* Bold Cyan */
#define BOLDWHITE   ((const char*)"\033[1m\033[37m")      /* Bold White */

static struct timeb timeb_s;
static char logbuffer[4096];

#define BT_SIZE  16
static inline void _BACKTRACE()
{
    int num_calls;
    void* bt_buffer[BT_SIZE];

    num_calls = backtrace(bt_buffer, BT_SIZE);

    backtrace_symbols_fd(bt_buffer, num_calls, STDERR_FILENO);
}
#undef BT_SIZE

static inline void _LOG_CORE(char level, struct timeb* timebp, pid_t pid,
                     const char* file,
                     int line,
                     const char* func,
                     char* msgstring)
{
    ftime(timebp);
    struct tm* ptm = localtime(&timebp->time);

    //Sets color to message
    fprintf (stderr,
               (level=='I') ? BOLDGREEN
             : (level=='W') ? BOLDYELLOW
             : (level=='E') ? BOLDRED
             : (level=='D') ? BOLDCYAN
             : 				  RESET);

    //actual message formatting
    fprintf (stderr, "[%1c%02d%02d %02d:%02d:%02d.%03d %05d %s:%d] %s : %s\n", level,
             ptm->tm_mon+1, ptm->tm_mday, ptm->tm_hour, ptm->tm_min, ptm->tm_sec,
             timebp->millitm, pid, file, line, func, msgstring);

    fprintf (stderr, RESET);
    memset(logbuffer, 0, sizeof(logbuffer));
}

//API
#define LOG_DEBUG(msg) do { \
_LOG_CORE ('D', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((msg))); \
} while (0)

#define LOG_INFO(msg) do { \
_LOG_CORE ('I', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((msg))); \
} while (0)

#define LOG_WARN(msg) do { \
_LOG_CORE ('W', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((msg))); \
} while (0)

#define LOG_ERROR(msg) do { \
_LOG_CORE ('E', &timeb_s, getpid(), __FILE__, __LINE__, __FUNCTION__, ((msg))); \
_BACKTRACE(); \
} while (0)

#define LOG_DEBUGF(...) do { \
snprintf (logbuffer, 4095, ##__VA_ARGS__); \
LOG_DEBUG (logbuffer); \
} while (0)

#define LOG_INFOF(...) do { \
snprintf (logbuffer, 4095, ##__VA_ARGS__); \
LOG_INFO (logbuffer); \
} while (0)

#define LOG_WARNF(...) do { \
snprintf (logbuffer, 4095, ##__VA_ARGS__); \
LOG_WARN (logbuffer); \
} while (0)

#define LOG_ERRORF(...) do { \
snprintf (logbuffer, 4095, ##__VA_ARGS__); \
LOG_ERROR (logbuffer); \
} while (0)

#endif
