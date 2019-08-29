#ifndef FAST_SLIC_TIMER_H
#define FAST_SLIC_TIMER_H
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <sstream>

namespace fstimer {
    typedef std::chrono::high_resolution_clock Clock;
    class TimerSection {
    private:
        std::string name;
        std::chrono::time_point<Clock> start_time;
        std::chrono::time_point<Clock> end_time;
        std::vector<std::unique_ptr<TimerSection>> children;
    public:
        TimerSection(const std::string& name) : name(name) {};
        void has_started() { start_time = Clock::now(); };
        void has_finished() {
            end_time = Clock::now();
        };

        void add_child(std::unique_ptr<TimerSection> child) {
            children.push_back(std::move(child));
        }

        template<typename T>
        long long get_duration() const {
            return std::chrono::duration_cast<T>(end_time - start_time).count();
        }

        long long get_nanos() const {
            return get_duration<std::chrono::nanoseconds>();
        };

        long long get_micros() const {
            return get_duration<std::chrono::microseconds>();
        };

        long long get_millis() const {
            return get_duration<std::chrono::milliseconds>();
        };

        std::vector<std::unique_ptr<TimerSection>>& get_children() {
            return children;
        }

        void dump_json(std::stringstream &stream);
    };

    class Timer {
    private:
        std::vector<std::unique_ptr<TimerSection>> stack;
        std::unique_ptr<TimerSection> last;
    public:
        Timer() {};
        void begin(const std::string& name);
        void end();
        std::string get_report();
    };
    void begin(const std::string& name);
    void end();
    std::string get_report();
    Timer& local_timer();

    class Scope {
    private:
        Timer &timer;
    public:
        Scope(Timer &timer, const std::string &name);
        Scope(const std::string &name);
        ~Scope();
    };
};

#endif
