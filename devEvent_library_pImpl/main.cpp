#include "library.h"
#include <iostream>

using real = float; // or double

int main() {

    float time;

    timer<false> hostTimer;
    hostTimer.tic();
    // Do some work on the host
    time = hostTimer.toc();
    std::cout << "Host time: " << time << std::endl;


    timer<true> devTimer;
    devTimer.tic();
    // Do some work on the device
    time = devTimer.toc();
    std::cout << "Device time: " << time << std::endl;

    return 0;
}