#pragma once

#include <functional>
#include <chrono>

/**
 * Util for time measurement based on RAII principe
 */
 template<typename Units>
class TimeMeasuring
{
public:
    using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;
    using OnFinishedCallback = std::function<void(int64_t)>;

public:
    TimeMeasuring(OnFinishedCallback&& onFinished) noexcept
            : _onFinished{std::move(onFinished)}
            , _startTime{std::chrono::steady_clock::now()}
    {
    }

    ~TimeMeasuring()
    {
       using namespace ::std::chrono;
       _onFinished(duration_cast<Units>(steady_clock::now() - _startTime).count());
    }

private:
    OnFinishedCallback _onFinished;
    TimePoint _startTime;
};

#define TAKEN_TIME_NS() auto _ = TimeMeasuring<::std::chrono::nanoseconds>([](uint64_t takenTime){ std::cout << __FILE__ << ":" << __LINE__ << ", taken: " << takenTime << "ns" << std::endl; })
#define TAKEN_TIME_US() auto _ = TimeMeasuring<::std::chrono::microseconds>([](uint64_t takenTime){ std::cout << __FILE__ << ":" << __LINE__ << ", taken: " << takenTime << "us" << std::endl; })
#define TAKEN_TIME_MS() auto _ = TimeMeasuring<::std::chrono::milliseconds>([](uint64_t takenTime){ std::cout << __FILE__ << ":" << __LINE__ << ", taken: " << takenTime << "ms" << std::endl; })