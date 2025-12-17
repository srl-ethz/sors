#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "common.h"

/**
 * @class Benchmark
 * @brief Static utility for collecting and reporting lightweight performance statistics.
 *
 * The Benchmark class aggregates timing measurements for named code regions. For each
 * name, it tracks:
 * - total accumulated time
 * - number of calls
 * - minimum and maximum duration
 * - mean and standard deviation of execution times
 *
 * Usage pattern:
 *   - Call Benchmark::enable() once before profiling.
 *   - Use ScopedTimer (via PROFILE_SCOPE / PROFILE_FUNCTION) in the code to record timings.
 *   - After the run, use print_report() for a formatted console summary or export_csv()
 *     to write the results to disk.
 *
 * All state is stored in static members and is thus global to the process.
 * Usage:
 * 
 *   Benchmark::enable();
 *   {
 *     PROFILE_SCOPE("My Operation");
 *     // ... code to profile ...
 *   }
 *   Benchmark::print_report();
 *   Benchmark::export_csv("results.csv");
 */
class Benchmark {
public:

    /**
     * @struct TimingStats
     * @brief Aggregates timing statistics for a single named operation.
     *
     * Stores raw sums and extrema of time measurements in microseconds and provides
     * convenience functions for computing derived statistics such as mean and
     * standard deviation. Used internally by Benchmark for each profiled label.
     */
    struct TimingStats {
        long long totalTime = 0;
        long long totalTimeSquared = 0;
        int callCount = 0;
        long long minTime = LLONG_MAX;
        long long maxTime = 0;

        // Updates statistics with a new timing measurement
        void update(long long time) {
            totalTime += time;
            totalTimeSquared += time * time;
            callCount++;
            minTime = std::min(minTime, time);
            maxTime = std::max(maxTime, time);
        }

        // Calculates average execution time
        double average() const {
            return callCount > 0 ? static_cast<double>(totalTime) / callCount : 0.0;
        }

        // Calculates standard deviation of execution times
        double stddev() const {
            if (callCount <= 1) return 0.0;
            double avg = average();
            double variance = static_cast<double>(totalTimeSquared) / callCount - avg * avg;
            return std::sqrt(std::max(0.0, variance));
        }
    };

    // Enables benchmark recording
    static void enable() { enabled_ = true; }

    // Disables benchmark recording
    static void disable() { enabled_ = false; }

    // Checks if benchmarking is currently enabled
    static bool is_enabled() { return enabled_; }

    // Clears all recorded timing data
    static void reset() { timings_.clear(); }

    // Records a timing measurement for a named operation
    static void record(const std::string& name, long long microseconds) {
        if (enabled_) {
            timings_[name].update(microseconds);
        }
    }

    // Prints formatted benchmark report to console
    static void print_report() {
        if (!enabled_ || timings_.empty()) return;

        std::cout << "\n" << bcolors.HEADER
                  << "============================== Performance Report =============================="
                  << bcolors.ENDC << "\n";
        std::cout << std::left << std::setw(35) << "Operation"
                  << std::right << std::setw(8) << "Calls"
                  << std::setw(12) << "Total(ms)"
                  << std::setw(11) << "Avg(ms)"
                  << std::setw(11) << "StdDev"
                  << std::setw(11) << "Min(ms)"
                  << std::setw(11) << "Max(ms)" << "\n";
        std::cout << std::string(99, '-') << "\n";

        for (const auto& [name, stats] : timings_) {
            std::cout << std::left << std::setw(35) << name
                      << std::right << std::setw(8) << stats.callCount
                      << std::setw(12) << std::fixed << std::setprecision(3)
                      << stats.totalTime / 1000.0
                      << std::setw(11) << stats.average() / 1000.0
                      << std::setw(11) << stats.stddev() / 1000.0
                      << std::setw(11) << stats.minTime / 1000.0
                      << std::setw(11) << stats.maxTime / 1000.0 << "\n";
        }
        std::cout << std::string(99, '=') << "\n\n";
    }

    // Exports benchmark results to CSV file
    static void export_csv(const std::string& filename) {
        if (!enabled_ || timings_.empty()) return;

        std::ofstream out(filename);
        if (!out) {
            std::cerr << bcolors.FAIL << "Failed to open " << filename
                      << " for writing" << bcolors.ENDC << std::endl;
            return;
        }

        out << "Operation,Calls,Total_ms,Avg_ms,StdDev_ms,Min_ms,Max_ms\n";
        for (const auto& [name, stats] : timings_) {
            out << name << ","
                << stats.callCount << ","
                << std::fixed << std::setprecision(3)
                << stats.totalTime / 1000.0 << ","
                << stats.average() / 1000.0 << ","
                << stats.stddev() / 1000.0 << ","
                << stats.minTime / 1000.0 << ","
                << stats.maxTime / 1000.0 << "\n";
        }

        std::cout << bcolors.OKGREEN << "Benchmark results exported to "
                  << filename << bcolors.ENDC << std::endl;
    }

    // Returns read-only access to all timing statistics
    static const std::map<std::string, TimingStats>& get_timings() {
        return timings_;
    }

private:
    static inline bool enabled_ = false;
    static inline std::map<std::string, TimingStats> timings_;
};

/**
 * @class ScopedTimer
 * @brief RAII-style helper that records elapsed time for the lifetime of a scope.
 *
 * On construction, captures the current time; on destruction, computes the elapsed
 * microseconds and forwards the measurement to Benchmark::record() under the
 * given name. Typically used through the PROFILE_SCOPE or PROFILE_FUNCTION macros.
 */
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        Benchmark::record(name_, elapsed.count());
    }
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

#define PROFILE_SCOPE(name) ScopedTimer timer(name)
#define PROFILE_FUNCTION() ScopedTimer timer(__FUNCTION__)

#endif // BENCHMARK_H