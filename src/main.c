#include <stdio.h>
#include <string.h>

extern int run_model_xor(int argc, char** argv);
extern int run_model_mnist(int argc, char** argv);

int main(int argc, char** argv)
{
    if(argc < 2) printf("usage: %s [xor | mnist]\n", argv[0]);
    else if(strcmp(argv[1], "mnist") == 0) run_model_mnist(argc, argv);
    else if(strcmp(argv[1], "xor") == 0) run_model_xor(argc, argv);
    else fprintf(stderr, "%s is not a valid option\n", argv[1]);
    return 0;
}
