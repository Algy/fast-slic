#include "timer.h"

namespace fstimer {
    void TimerSection::dump_json(std::stringstream &stream) {
        stream << "{\"name\": \"" << name << "\", \"duration\": " << get_micros()
            << ", \"children\": ";
        stream << "[";
        bool is_first = true;
        for (const auto & child : children) {
            if (is_first) {
                is_first = false;
            } else {
                stream << ",";
            }
            child->dump_json(stream);
        }
        stream << "]}";
    }

    void Timer::begin(const std::string &name) {
        std::unique_ptr<TimerSection> section { new TimerSection(name) };
        section->has_started();
        stack.push_back(std::move(section));
    }

    void Timer::end() {
        if (stack.empty()) return;
        std::unique_ptr<TimerSection> section { std::move(stack.back()) };
        stack.pop_back();
        section->has_finished();
        if (stack.empty()) {
            last = std::move(section);
        } else {
            stack.back()->add_child(std::move(section));
        }
    }

    std::string Timer::get_report() {
        if (last == nullptr) return std::string("");
        std::stringstream stream;
        last->dump_json(stream);
        return stream.str();
    }

    thread_local Timer thread_local_timer;
    Timer& local_timer() { return thread_local_timer; };
    void begin(const std::string& name) { thread_local_timer.begin(name); };
    void end() { thread_local_timer.end(); };
    std::string get_report() { return thread_local_timer.get_report(); };

    Scope::Scope(Timer &timer, const std::string &name) : timer(timer) { timer.begin(name); };
    Scope::Scope(const std::string &name) : Scope(thread_local_timer, name)  {};
    Scope::~Scope() { timer.end(); };
};
