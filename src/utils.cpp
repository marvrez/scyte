#include "utils.h"

#include <chrono>

double time_now() 
{
    using namespace std::chrono;
    nanoseconds now = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
    return now.count() * 1e-9;
}
