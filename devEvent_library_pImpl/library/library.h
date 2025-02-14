#ifndef LIBRARY_H
#define LIBRARY_H

// My "super" library for measuring time on GPU and CPU

#include <chrono>
#include <memory>


class DeviceEventImpl; // Forward declaration


class DeviceEvent {
  public:
    DeviceEvent();
    ~DeviceEvent();

    void record();
    void stop();
    float elapsed();

  private:
    std::unique_ptr<DeviceEventImpl> pImpl; // Pointer to implementation
};


class HostEvent {
  public:
    HostEvent() = default;
    ~HostEvent() = default;

    void record() 
    {
      start = std::chrono::high_resolution_clock::now();
    }

    void stop() 
    {
      end = std::chrono::high_resolution_clock::now();
    }

    float elapsed() 
    {
      const std::chrono::duration<float> elapsed = end - start;
      return elapsed.count();
    }

  private:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
};


template <bool IS_ON_DEV = false>
class timer {
  public:
    timer() : event() {}
    ~timer() = default;

    void tic() {
      event.record();
    }

    float toc() {
      event.stop();
      return event.elapsed();
    }

  private:
    std::conditional_t<IS_ON_DEV, DeviceEvent, HostEvent> event;
};


#endif // LIBRARY_H