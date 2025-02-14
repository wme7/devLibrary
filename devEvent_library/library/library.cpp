#include "library.h"
#include "common.h"


DeviceEvent::DeviceEvent() {
    DEV_CHECK(devEventCreate(&start));
    DEV_CHECK(devEventCreate(&end));
}

DeviceEvent::~DeviceEvent() {
    DEV_CHECK(devEventDestroy(start));
    DEV_CHECK(devEventDestroy(end));
}

void DeviceEvent::record() {
    DEV_CHECK(devEventRecord(start, 0));
}

void DeviceEvent::stop() {
    DEV_CHECK(devEventRecord(end, 0));
    DEV_CHECK(devEventSynchronize(end));
}

float DeviceEvent::elapsed() {
    float milliseconds;
    DEV_CHECK(devEventElapsedTime(&milliseconds, start, end));
    return 0.001 * milliseconds;
}
