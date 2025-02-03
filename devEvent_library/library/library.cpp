#include "library.h"
#include "common.h"

// Using the PIMPL c++ idiom

class DeviceEventImpl {
  public:
    DeviceEventImpl() {
        DEV_CHECK(devEventCreate(&start));
        DEV_CHECK(devEventCreate(&end));
    }

    ~DeviceEventImpl() {
        DEV_CHECK(devEventDestroy(start));
        DEV_CHECK(devEventDestroy(end));
    }

    void record() {
        DEV_CHECK(devEventRecord(start, 0));
    }

    void stop() {
        DEV_CHECK(devEventRecord(end, 0));
        DEV_CHECK(devEventSynchronize(end));
    }

    float elapsed() {
        float milliseconds;
        DEV_CHECK(devEventElapsedTime(&milliseconds, start, end));
        return 0.001 * milliseconds;
    }

  private:
    devEvent_t start;
    devEvent_t end;
};

DeviceEvent::DeviceEvent() : pImpl(std::make_unique<DeviceEventImpl>()) {}

DeviceEvent::~DeviceEvent() = default;

void DeviceEvent::record() {
    pImpl->record();
}

void DeviceEvent::stop() {
    pImpl->stop();
}

float DeviceEvent::elapsed() {
    return pImpl->elapsed();
}