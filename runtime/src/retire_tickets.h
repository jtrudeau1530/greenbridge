#pragma once

#include <cstdint>
#include <deque>

namespace leonida {

// Global submission tickets may finish out of order on independent Vulkan queues. Only the
// contiguous completed prefix is safe for deferred resource destruction. The caller serializes
// record(), completed() and queue_idle() with the device submission mutex.
template<class Semaphore>
class RetireTickets {
public:
    void record(std::uint64_t ticket, Semaphore semaphore, std::uint64_t value) {
        pending_.push_back({ticket, semaphore, value, false});
    }

    template<class Query>
    std::uint64_t completed(Query&& query) {
        while (!pending_.empty()) {
            const Entry& front = pending_.front();
            if (!front.idle && !query(front.semaphore, front.value)) break;
            completed_ = front.ticket;
            pending_.pop_front();
        }
        return completed_;
    }

    // The owning queue has been waited idle and may now destroy its progress semaphore.
    void queue_idle(Semaphore semaphore) {
        for (Entry& entry : pending_)
            if (entry.semaphore == semaphore) entry.idle = true;
    }

private:
    struct Entry {
        std::uint64_t ticket;
        Semaphore semaphore;
        std::uint64_t value;
        bool idle;
    };
    std::deque<Entry> pending_;
    std::uint64_t completed_ = 0;
};

}  // namespace leonida
