#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cuda_runtime.h>
#include <functional>

struct PageKey{
    int layer_id;
    int expert_id;
    int version;
    bool operator==(const PageKey& other) const {
        return layer_id == other.layer_id && 
        expert_id == other.expert_id && version == other.version;
    }
};
struct PageKeyHash{
    std::size_t operator()(const PageKey& k) const {
        return ((std::hash<int>()(k.layer_id) 
        ^ (std::hash<int>()(k.expert_id) << 1)));
    }
};
class MoEManager {
    public : 
        // Generator
        MoEManager(int max_gpu_slots, size_t slot_size, int num_layers);
        ~MoEManager();
        // Core APIs
        // ???
        void register_expert(int layer_id, int expert_id, size_t size_bytes, long long cpu_ptr,
        int version);
        // ???
        long long request_expert(int layer_id, int expert_id, int version);
        // don't need it
        // void prefetch_expert(int layer_id, int expert_id, int version);
        // kick out experts
        int evict_one();

        // ???
        void lock_expert(int layer_id, int expert_id, int version);
        void unlock_expert(int layer_id, int expert_id, int version);
        void wait_for_transfer(long long stream_ptr);
        void transfer_waits_for_compute(long long stream_ptr);
        void print_stats();

    private:

        struct ExpertMeta {
            PageKey key;
            size_t size_bytes;
            long long cpu_ptr;
            int gpu_slot_idx;
            bool is_resident;
            bool referenced;
        };

        void maintain_free_pool();

        int max_gpu_slots;
        size_t slot_size;
        int num_layers;

        int min_free_threshold;

        std::vector<void*> gpu_slots;
        std::unordered_map<PageKey, ExpertMeta, PageKeyHash> page_table;
        std::vector<PageKey> slot_owner;
        std::unordered_set<PageKey, PageKeyHash> locked_experts;

        std::vector<int> layer_usage;

        int clock_hand;
        int free_slots_count;

        cudaStream_t transfer_stream;
        cudaEvent_t transfer_done_event;
        cudaEvent_t compute_done_event;

        long long total_requests = 0;
        long long cache_hits = 0;
        long long cache_misses = 0;
        long long eviction_count = 0;
};