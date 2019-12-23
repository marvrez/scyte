#ifndef ARG_H
#define ARG_H

#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ARGS 255
typedef unsigned int arg_flag;
typedef enum { arg_int=0, arg_float, arg_string, } arg_type;

/* values for flags: */
#define ARG_OPTIONAL  0x000 /* option may or may not have an option-argument */
#define ARG_FORBIDDEN 0x001 /* option must not have an option-argument */
#define ARG_REQUIRED  0x002 /* option must have an option-argument */
#define ARG_NO_HYPHEN 0x004 /* option-argument may not start with hyphen/minus */
#define ARG_REPEATABLE         0x008 /* option may be specified more than once */
#define ARG_SEEN_SHORT_WITHOUT 0x010 /* short form of option was present without an option-argument in argv */
#define ARG_SEEN_SHORT_WITH    0x020 /* short form of option was present with an option-argument in argv */
#define ARG_SEEN_LONG_WITHOUT  0x040 /* long form of option was present without an option-argument in argv */
#define ARG_SEEN_LONG_WITH     0x080 /* long form of option was present with an option-argument in argv */

typedef struct {
  char          short_name;
  const char*   long_name;
  const char*   help;

  arg_type      type;
  arg_flag      flags;

  unsigned int  count;
  void*         arg;
} arg_option;

static struct { int num_opts; arg_option options[MAX_ARGS+1]; } global_options = { 0 };

#define arg_add_option(SUFFIX, TYPE) \
    static inline void arg_option_##SUFFIX(TYPE* arg, char short_name, const char* long_name, const char* help, arg_flag flags) \
    { \
        arg_option opt = { short_name, long_name, help, arg_##SUFFIX, flags, 0, arg }; \
        assert(global_options.num_opts < MAX_ARGS); \
        global_options.options[global_options.num_opts++] = opt; \
    }

arg_add_option(int, int);
arg_add_option(float, float);
arg_add_option(string, const char*);

static void arg_option_count(int* count, char short_name, const char* long_name, const char* help)
{
    arg_option_int(count, short_name, long_name, help, ARG_FORBIDDEN);
}

#undef arg_add_option

static inline unsigned int get_short_option_index(char c)
{
    unsigned int i;
    for(i = 0; i < global_options.num_opts; ++i) {
        if(global_options.options[i].short_name == c) return i;
    }
    return i;
}

static inline unsigned int get_long_option_index(const char* arg)
{
    unsigned int count = 0, found, i;
    size_t arg_len = strcspn(arg, "=");
    for(i = 0; i < global_options.num_opts; ++i) {
        if(global_options.options[i].long_name) {
            size_t full_len = strlen(global_options.options[i].long_name);
            if(arg_len <= full_len && !strncasecmp(global_options.options[i].long_name, arg, arg_len)) {
                if(arg_len == full_len) return i;
                found = i, count++;
            }
        }
    }
    return count == 1 ? found : i;
}

static inline void set_arg_val(arg_option* opt, char* val)
{
    if(opt->type == arg_int) {
        int* arg = (int*)opt->arg;
        *arg = atoi(val);
    }
    else if (opt->type == arg_float) {
        float* arg = (float*)opt->arg;
        *arg = atof(val);
    }
    else if (opt->type == arg_string) {
        char** arg = (char**)opt->arg;
        *arg = val;
    }
}

static void arg_check_errors(const char* argv0)
{
    arg_option* options = global_options.options;
    int num_opts = global_options.num_opts;

    if(options[num_opts].short_name) {
        fprintf(stderr, "%s: unrecognised option -%c\n", argv0, options[num_opts].short_name);
        exit(EXIT_FAILURE);
    }
    if(options[num_opts].long_name) {
        fprintf(stderr, "%s: unrecognised option --%s\n", argv0, options[num_opts].long_name);
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < num_opts; ++i) {
        if((options[i].flags & ARG_SEEN_SHORT_WITHOUT) && (options[i].flags & ARG_REQUIRED)) {
            fprintf(stderr, "%s: option -%c requires an option-argument\n", argv0, options[i].short_name);
            exit(EXIT_FAILURE);
        }
        if((options[i].flags & ARG_SEEN_SHORT_WITH) && (options[i].flags & ARG_FORBIDDEN)) {
            fprintf(stderr, "%s: option -%c must not have an option-argument\n", argv0, options[i].short_name);
            exit(EXIT_FAILURE);
        }
        if((options[i].flags & ARG_SEEN_LONG_WITHOUT) && (options[i].flags & ARG_REQUIRED)) {
            fprintf(stderr, "%s: option --%s requires an option-argument\n", argv0, options[i].long_name);
            exit(EXIT_FAILURE);
        }
        if((options[i].flags & ARG_SEEN_LONG_WITH) && (options[i].flags & ARG_FORBIDDEN)) {
            fprintf(stderr, "%s: option --%s must not have an option-argument\n", argv0, options[i].long_name);
            exit(EXIT_FAILURE);
        }
        if((options[i].count > 1) && !(options[i].flags & ARG_REPEATABLE)) {
            fprintf(stderr, "%s: option -%c/--%s may not be repeated\n", argv0, options[i].short_name, options[i].long_name);
            exit(EXIT_FAILURE);
        }
    }
}

static int arg_parse(char** argv)
{
    unsigned int operand_count = 1, doubledash = 0, expecting = 0;
    unsigned int opt_idx, i, j;
    arg_option* options = global_options.options;
    for(i = 1; argv[i]; ++i) {
        if(doubledash) {
            argv[operand_count++] = argv[i];
            continue;
        }
        if(expecting) {
            if((argv[i][0] != '-') || !argv[i][1] || !(options[opt_idx].flags & ARG_NO_HYPHEN)) {
                options[opt_idx].flags |= expecting;
                set_arg_val(&options[opt_idx], argv[i]);
                expecting = 0;
                continue;
            }
            else {
                options[opt_idx].flags |= (expecting >> 1); /* change WITH to WITHOUT */
                options[opt_idx].arg = NULL;
                expecting = 0;
            }
        }

        if(argv[i][0] == '-' && argv[i][1] == '-' && argv[i][2] == 0) {
            doubledash = 1;
            continue;
        }

        if(argv[i][0] == '-' && argv[i][1] == '-') {
            char* eq = strchr(&argv[i][2], '=');
            opt_idx = get_long_option_index(&argv[i][2]);
            options[opt_idx].count++;
            if(opt_idx == global_options.num_opts && !options[opt_idx].long_name) options[opt_idx].long_name = &argv[i][2];

            if(eq) {
                set_arg_val(&options[opt_idx], eq + 1);
                options[opt_idx].flags |= ARG_SEEN_LONG_WITH;
            }
            else if(options[opt_idx].flags & ARG_REQUIRED) expecting = ARG_SEEN_LONG_WITH;
            else {
                options[opt_idx].arg = NULL;
                options[opt_idx].flags |= ARG_SEEN_LONG_WITHOUT;
            }
        }
        else if(argv[i][0] == '-' && argv[i][1]) {
            for(j = 1; argv[i][j]; ++j) {
                opt_idx = get_short_option_index(argv[i][j]);
                options[opt_idx].count++;

                if(opt_idx == global_options.num_opts) {
                    if(!options[opt_idx].short_name) options[opt_idx].short_name = argv[i][j];
                    if(argv[i][j+1]) {
                        set_arg_val(&options[opt_idx], &argv[i][j+1]);
                        options[opt_idx].flags |= ARG_SEEN_SHORT_WITH;
                    }
                    else {
                        options[opt_idx].arg = NULL;
                        options[opt_idx].flags |= ARG_SEEN_SHORT_WITHOUT;
                    }
                    break;
                }

                if(options[opt_idx].flags & ARG_FORBIDDEN) {
                    *(int*)options[opt_idx].arg = (int)options[opt_idx].count,
                    options[opt_idx].arg = NULL;
                    options[opt_idx].flags |= ARG_SEEN_SHORT_WITHOUT;
                }
                else if(argv[i][j+1]) {
                    set_arg_val(&options[opt_idx], &argv[i][j+1]);
                    options[opt_idx].flags |= ARG_SEEN_SHORT_WITH;
                    break;
                }
                else if(options[opt_idx].flags & ARG_REQUIRED) expecting = ARG_SEEN_SHORT_WITH;
                else {
                    options[opt_idx].arg = NULL;
                    options[opt_idx].flags |= ARG_SEEN_SHORT_WITHOUT;
                }
            }
        }
        else argv[operand_count++] = argv[i];
    }

    if(expecting) {
        options[opt_idx].flags |= expecting >> 1; // change _WITH to _WITHOUT
        options[opt_idx].arg  =  NULL;
    }

    argv[operand_count] = NULL;
    arg_check_errors(argv[0]);
    return operand_count;
}

static void arg_help()
{
    printf("Options:\n");
    for(int i = 0; i < global_options.num_opts; ++i) {
        arg_option opt = global_options.options[i];
        const char* type_string;
        switch(opt.type) {
            case arg_int: type_string = "INT"; break;
            case arg_float: type_string = "FLOAT"; break;
            case arg_string: type_string = "STRING"; break;
        }
        if(opt.flags & ARG_FORBIDDEN) type_string = "";
        char short_name_str[256] = {""}, long_name_str[256] = {""};
        if(opt.short_name) sprintf(short_name_str, "-%c", opt.short_name);
        if(opt.long_name) sprintf(long_name_str, "--%s", opt.long_name);
        printf("    %-3s %-20s%-10s     %s\n", short_name_str, long_name_str, type_string, opt.help);
    }
}

#endif
