#include "utils.h"

#include <chrono>

double time_now() 
{
    using namespace std::chrono;
    seconds now = duration_cast<seconds>(system_clock::now().time_since_epoch());
    return now.count();
}
