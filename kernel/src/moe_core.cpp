#include "moe_core.hpp"
#include <iostream>
#include <algorithm>
#include <stdexcept>

static const PageKey NULL_KEY = {-1, -1, -1};

MoEManager::MoEManager(int max_gpu_slots, size_t slot_size, int num_layers)
    : max_gpu_slots(max_gpu_slots), slot_size(slot_size), 
    num_layers(num_layers), slot_owner(max_gpu_slots, NULL_KEY),
    clock_hand(0),
    free_slots_count(max_gpu_slots)
{
    // for pressure, Why we need this ?
    // min_free_threshold = std::max(1, (int)(max_gpu_slots * 0.05));
    
    // Quota, we need to track usage and quoted
    layer_usage.resize(num_layers, 0);

    // cuda async stream and event for track compututation and PCI
    cudaStreamCreateWithFlags(&transfer_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&transfer_done_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&compute_done_event, cudaEventDisableTiming);
    // malloc gpu slots
    gpu_slots.resize(max_gpu_slots);
    for (int i = 0; i < max_gpu_slots; ++i){
        if (cudaMalloc(&gpu_slots[i], slot_size) != cudaSuccess){
            throw std::runtime_error("CUDA Malloc failed");
        }
    }
};
// Delete pinned tensors and stream, events
MoEManager::~MoEManager(){
    cudaEventDestroy(transfer_done_event);
    cudaEventDestroy(compute_done_event);
    cudaStreamDestroy(transfer_stream);
    for (auto ptr : gpu_slots) cudaFree(ptr); 
}

// Background Evictor
void MoEManager::maintain_free_pool(){
    int safeguard = 0;
    while (free_slots_count < min_free_threshold && 
    safeguard < max_gpu_slots){
        if (evict_one() == -1) break;
        safeguard ++;
    }
}

// Register
void MoEManager::register_expert(int layer_id, int expert_id, size_t size_bytes,
long long cpu_ptr, int version){
    PageKey key = {layer_id, expert_id, version};
    ExpertMeta meta;
    meta.key = key;
    meta.size_bytes = size_bytes;
    meta.cpu_ptr = cpu_ptr;
    meta.gpu_slot_idx = -1;
    meta.is_resident = false;
    meta.referenced = false;
    page_table[key] = meta;
}

//Lock
void MoEManager::lock_expert(int layer_id, int expert_id, int version){
    locked_experts.insert({layer_id, expert_id, version});
}
//Unlock
void MoEManager::unlock_expert(int layer_id, int expert_id, int version){
    locked_experts.erase({layer_id, expert_id, version});
}

long long MoEManager::request_expert(int layer_id, int expert_id, int version){
    total_requests++;
    // maintain_free_pool();

    PageKey key = {layer_id, expert_id, version};
    if (page_table.find(key) == page_table.end()){
        throw std::runtime_error("Invalid Expert Request");
    }
    ExpertMeta& meta = page_table[key];

    //HIT
    if (meta.is_resident){
        cache_hits++;
        meta.referenced = true; // Second wind
        return (long long)gpu_slots[meta.gpu_slot_idx]; // return here
    }
    // Miss
    cache_misses++;
    // find free slots
    int target_slot = -1;
    for (int i = 0; i < max_gpu_slots; ++i) {
        if (slot_owner[i].layer_id == -1) { target_slot = i; break; }
    }

    // If no free slots then evicate
    if (target_slot == -1) target_slot = evict_one();
    // if evicate failed then shut down
    if (target_slot == -1) throw std::runtime_error("OOM: Eviction Failed (All locked?)");
    
    cudaMemcpyAsync(gpu_slots[target_slot], (void*)meta.cpu_ptr,
    meta.size_bytes, cudaMemcpyHostToDevice, transfer_stream);

    meta.gpu_slot_idx = target_slot;
    meta.is_resident = true;
    meta.referenced = true;

    if (layer_id < num_layers) layer_usage[layer_id]++;
    free_slots_count --;

    slot_owner[target_slot] = key;
    return (long long)gpu_slots[target_slot];
}

// Evicate
int MoEManager::evict_one(){
    int checked_count = 0;
    int total_slots = max_gpu_slots;

    while (checked_count < total_slots * 2){
        int candidate_slot = clock_hand;
        PageKey owner_key = slot_owner[candidate_slot];

        clock_hand = (clock_hand + 1) % total_slots;
        checked_count++;

        if (owner_key.layer_id == -1) return candidate_slot;
        // Skip if the expert is locked
        if (locked_experts.find(owner_key) != locked_experts.end()) continue;

        ExpertMeta& meta = page_table[owner_key];

        // CLOCK Logic
        if (meta.referenced){
            meta.referenced = false; // Second wind
        }else {
            // Victim Selected
            if (owner_key.layer_id < num_layers) layer_usage[owner_key.layer_id]--;
            
            meta.is_resident = false;
            meta.gpu_slot_idx = -1;
            slot_owner[candidate_slot] = NULL_KEY;
            free_slots_count++;
            eviction_count++;
            
            return candidate_slot;
            }
        }
    return -1;
}
// Sync Function for wait transfer
void MoEManager::wait_for_transfer(long long stream_ptr){
    cudaEventRecord(transfer_done_event, transfer_stream);
    cudaStream_t torch_stream = (cudaStream_t)stream_ptr;
    cudaStreamWaitEvent(torch_stream, transfer_done_event, 0);

}
// Sync Function for wait compute
void MoEManager::transfer_waits_for_compute(long long stream_ptr) {
    cudaStream_t torch_stream = (cudaStream_t)stream_ptr;
    cudaEventRecord(compute_done_event, torch_stream);
    cudaStreamWaitEvent(transfer_stream, compute_done_event, 0);
}



void MoEManager::print_stats() {
    std::cout << "[MoE-Manager] Req: " << total_requests 
              << " | Hit: " << cache_hits 
              << " | Miss: " << cache_misses 
              << " | Evicts: " << eviction_count 
              << " | Free: " << free_slots_count << "\n";
    
    std::cout << "\n";
}