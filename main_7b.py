import time
import random
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import libmoe

def print_moe_cache_stats(model):
    if hasattr(model, "moe_pager"):
        model.moe_pager.manager.print_stats()
        
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
        
class UnsafeGPUTensorWrapper:
    def __init__(self, ptr, shape, dtype):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
    @property
    def __cuda_array_interface__(self):
        typestr = np.dtype(str(self.dtype).replace("torch.", "")).str
        return {"shape": self.shape, "typestr": typestr, "data": (self.ptr, False), "version": 3}

class GlobalExpertPager:
    def __init__(self, all_layers_experts, max_global_slots, device):
        self.device = device
        self.tensor_cache = {}
        self.hot_experts = None
        self.pinned_memory_refs = []
        
        num_layers = len(all_layers_experts)

        print("[System] Aligning & PINNING Expert Weights...")
        self.all_layers_experts = []
        for layer_experts in tqdm(all_layers_experts, desc="Processing Layers"):
            new_layer = []
            for exp in layer_experts:
                w1_p = exp.w1.weight.data.contiguous().pin_memory()
                w2_p = exp.w2.weight.data.contiguous().pin_memory()
                w3_p = exp.w3.weight.data.contiguous().pin_memory()
                self.pinned_memory_refs.extend([w1_p, w2_p, w3_p])
                
                exp.w1.weight.data = w1_p
                exp.w2.weight.data = w2_p
                exp.w3.weight.data = w3_p
                new_layer.append(exp)
            self.all_layers_experts.append(new_layer)

        sample_exp = self.all_layers_experts[0][0]
        self.w1_shape = sample_exp.w1.weight.shape
        self.w2_shape = sample_exp.w2.weight.shape
        self.w3_shape = sample_exp.w3.weight.shape
        self.dtype = sample_exp.w1.weight.dtype
        element_size = 2 if self.dtype == torch.float16 else 4

        slot_size = sample_exp.w1.weight.numel() * element_size
        self.total_slots = max_global_slots * 3

        print(f"[System] Global VRAM Pool: {max_global_slots} Experts Capacity (Slots: {self.total_slots})")
        
        # [Update] num_layers 인자 전달 (Quota 관리용)
        self.manager = libmoe.MoEManager(self.total_slots, slot_size, num_layers)

        # [Update] (Layer, Expert, Version) 구조로 등록
        # Version 0: W1, Version 1: W2, Version 2: W3
        for layer_idx, layer_exps in enumerate(self.all_layers_experts):
            for exp_idx, exp in enumerate(layer_exps):
                self.manager.register_expert(layer_idx, exp_idx, slot_size, exp.w1.weight.data_ptr(), 0)
                self.manager.register_expert(layer_idx, exp_idx, slot_size, exp.w2.weight.data_ptr(), 1)
                self.manager.register_expert(layer_idx, exp_idx, slot_size, exp.w3.weight.data_ptr(), 2)

    def _wrap_tensor(self, ptr: int, shape, dtype):
        key = (ptr, shape)
        if key in self.tensor_cache: return self.tensor_cache[key]
        wrapper = UnsafeGPUTensorWrapper(ptr, shape, dtype)
        t = torch.as_tensor(wrapper, device=self.device)
        self.tensor_cache[key] = t
        return t

    def prefetch(self, layer_idx, exp_idx):
        # [Update] 3개 버전 모두 Prefetch
        self.manager.request_expert(layer_idx, exp_idx, 0)
        self.manager.request_expert(layer_idx, exp_idx, 1)
        self.manager.request_expert(layer_idx, exp_idx, 2)

    def get_tensors(self, layer_idx, exp_idx):
        # [Update] 명시적 버전 요청
        ptr_w1 = self.manager.request_expert(layer_idx, exp_idx, 0)
        ptr_w2 = self.manager.request_expert(layer_idx, exp_idx, 1)
        ptr_w3 = self.manager.request_expert(layer_idx, exp_idx, 2)
        
        w1 = self._wrap_tensor(ptr_w1, self.w1_shape, self.dtype)
        w2 = self._wrap_tensor(ptr_w2, self.w2_shape, self.dtype)
        w3 = self._wrap_tensor(ptr_w3, self.w3_shape, self.dtype)
        return w1, w2, w3
    
    def lock_single_expert(self, layer_idx, exp_idx):
        self.manager.lock_expert(layer_idx, exp_idx, 0)
        self.manager.lock_expert(layer_idx, exp_idx, 1)
        self.manager.lock_expert(layer_idx, exp_idx, 2)

    def unlock_single_expert(self, layer_idx, exp_idx):
        self.manager.unlock_expert(layer_idx, exp_idx, 0)
        self.manager.unlock_expert(layer_idx, exp_idx, 1)
        self.manager.unlock_expert(layer_idx, exp_idx, 2)
        
class PagedMixtralSparseMoeBlock(nn.Module):
    def __init__(self, original_module, pager, layer_idx):
        super().__init__()
        self.hidden_dim = original_module.hidden_dim
        self.ffn_dim = original_module.ffn_dim
        self.num_experts = original_module.num_experts
        self.top_k = original_module.top_k
        self.gate = original_module.gate
        self.pager = pager
        self.layer_idx = layer_idx
        self.chunk_capacity = 4
    def forward(self, hidden_states:torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        unique_experts = torch.unique(selected_experts).tolist()
        
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )       
        chunks = [unique_experts[i : i + self.chunk_capacity] for i in range(0, len(unique_experts), self.chunk_capacity)]
        current_stream_ptr = torch.cuda.current_stream().cuda_stream
        
        for i, current_chunk in enumerate(chunks):
            # 1. Pipeline Sync: Wait for previous Compute
            self.pager.manager.transfer_waits_for_compute(current_stream_ptr)

            # 2. Lock Experts for this chunk (prevent eviction while computing)
            for eid in current_chunk:
                self.pager.prefetch(self.layer_idx, eid)
            # 3. Request / Get Tensors (Triggers DMA if MISS)
            current_tensors = {}
            for eid in current_chunk:
                w1, w2, w3 = self.pager.get_tensors(self.layer_idx, eid)
                current_tensors[eid] = (w1, w2, w3)

            # 4. Prefetch Next Chunk (Pipeline overlap)
            if i + 1 < len(chunks):
                for eid in chunks[i + 1]:
                    self.pager.prefetch(self.layer_idx, eid)

            # 5. Pipeline Sync: Wait for DMA to finish
            self.pager.manager.wait_for_transfer(current_stream_ptr)

            # 6. Compute
            for eid in current_chunk:
                expert_mask = selected_experts == eid
                expert_indices = torch.where(expert_mask)
                if len(expert_indices[0]) == 0: continue

                curr_state = hidden_states[expert_indices[0]]
                w1, w2, w3 = current_tensors[eid]
                
                res = F.silu(F.linear(curr_state, w1)) * F.linear(curr_state, w3)
                res = F.linear(res, w2)
                
                rw = routing_weights[expert_indices[0], expert_indices[1]].unsqueeze(-1)
                final_hidden_states.index_add_(0, expert_indices[0], res * rw)

            # 7. Unlock Experts (Allow eviction via CLOCK/Quota)
            for eid in current_chunk:
                self.pager.unlock_single_expert(self.layer_idx, eid)

        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits
       
def load_and_patch_system(model_id, max_global_slots):
    print(f"--- [System] Loading Model... ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    
    all_layers_experts = [[exp for exp in layer.block_sparse_moe.experts] for layer in model.model.layers]
    global_pager = GlobalExpertPager(all_layers_experts, max_global_slots, "cuda:0")
    
    # [OS-Cosplay] Quota 설정 예시 (선택 사항)
    # 레이어 0~3은 문법적인 부분이라 너무 자주 접근됨 -> Quota 제한으로 독점 방지
    print("--- [System] Patching Layers... ---")
    for i, layer in enumerate(model.model.layers):
        old_moe = layer.block_sparse_moe
        layer.block_sparse_moe = PagedMixtralSparseMoeBlock(old_moe, global_pager, i)
        del old_moe.experts
        del old_moe

    gc.collect()
    torch.cuda.empty_cache()
    model.to("cuda:0")
    model.eval()
    
    model.moe_pager = global_pager
    return model, tokenizer

import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

def eval_choice_likelihood(model, tokenizer, device, inputs, prefix_len):
    """
    Core logic from your snippet: calculates loss for the target segment.
    """
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Shift for autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attn   = attention_mask[..., 1:].contiguous()

        seq_len_shift = shift_labels.size(1)
        pos = torch.arange(seq_len_shift, device=device)
        
        # Mask out the prefix (Question/Context) so we only score the Answer
        answer_pos_mask = (pos >= (prefix_len - 1)).unsqueeze(0)
        answer_mask = (answer_pos_mask & (shift_attn == 1))

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        token_losses = token_losses.view_as(shift_labels)
        token_losses = token_losses * answer_mask.float()

        sum_loss = token_losses.sum(dim=1)
        num_tokens = answer_mask.sum(dim=1)

        # Fallback for empty targets (rare edge case)
        fallback_mask = (num_tokens == 0)
        if fallback_mask.any():
            non_pad_mask = (shift_attn == 1)
            fb_sum = (loss_fct(
                shift_logits[fallback_mask].view(-1, shift_logits.size(-1)),
                shift_labels[fallback_mask].view(-1),
            )).view(-1, seq_len_shift)
            fb_sum = (fb_sum * non_pad_mask[fallback_mask].float()).sum(dim=1)
            fb_cnt = non_pad_mask[fallback_mask].sum(dim=1)
            sum_loss[fallback_mask] = fb_sum
            num_tokens[fallback_mask] = fb_cnt

        seq_losses = sum_loss / num_tokens.clamp(min=1)
        return seq_losses.cpu().numpy()

def evaluate_model(model, tokenizer, device_name, is_system_mode=False, num_samples=50, tasks=None):
    """
    Extended evaluation function supporting: CSQA, AGN, HS, MMLU, MNLI, MRPC, WIN
    """
    device = "cuda" if "cuda" in device_name else "cpu"
    
    # Default to all if not specified
    if tasks is None:
        tasks = ["CSQA", "AGN", "HS", "MMLU", "MNLI", "MRPC", "Win"]

    results = {}

    for task in tasks:
        print(f"\n[Evaluation] Running {task} (Mode: {device_name})...")
        
        # --- 1. Dataset Loading & Formatting ---
        if task == "CSQA":
            dataset = load_dataset("tau/commonsense_qa", split="validation", trust_remote_code=True).select(range(num_samples))
            label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            
            def format_fn(ex):
                q = f"Question: {ex['question']}\nAnswer:"
                opts = ex["choices"]["text"]
                ans = label_map[ex["answerKey"]]
                return q, opts, ans

        elif task == "AGN": # AG News (Topic Classification)
            dataset = load_dataset("ag_news", split="test", trust_remote_code=True).select(range(num_samples))
            # 0:World, 1:Sports, 2:Business, 3:Sci/Tech
            options = ["World", "Sports", "Business", "Technology"] 
            
            def format_fn(ex):
                q = f"Classify the following text into a category.\nText: {ex['text']}\nCategory:"
                ans = ex["label"]
                return q, options, ans

        elif task == "HS": # HellaSwag (Commonsense Completion)
            dataset = load_dataset("rowan/hellaswag", split="validation", trust_remote_code=True).select(range(num_samples))
            
            def format_fn(ex):
                # HellaSwag provides Context A and B. We join them as prefix.
                q = f"{ex['ctx_a']} {ex['ctx_b']}" 
                # HellaSwag endings are raw continuations, no extra formatting needed usually
                opts = ex["endings"]
                ans = int(ex["label"])
                return q, opts, ans

        elif task == "MMLU": # Massive Multitask Language Understanding
            # Using a sample subset (e.g., general knowledge) or 'all' if handling large data
            try:
                dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True).select(range(num_samples))
            except:
                # Fallback if 'all' is too heavy or config requires specific name
                dataset = load_dataset("cais/mmlu", "global_facts", split="test", trust_remote_code=True).select(range(num_samples))
            
            label_map = {0: 0, 1: 1, 2: 2, 3: 3} # A, B, C, D maps to index 0, 1, 2, 3
            
            def format_fn(ex):
                q = f"{ex['question']}\nAnswer:"
                opts = ex["choices"]
                ans = ex["answer"]
                return q, opts, ans

        elif task == "MNLI": # Multi-Genre NLI (Entailment)
            dataset = load_dataset("glue", "mnli", split="validation_matched", trust_remote_code=True).select(range(num_samples))
            options = ["entailment", "neutral", "contradiction"]
            
            def format_fn(ex):
                # Zero-shot NLI template
                q = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nPrediction:"
                ans = ex["label"]
                return q, options, ans

        elif task == "MRPC": # Paraphrase Detection
            dataset = load_dataset("glue", "mrpc", split="validation", trust_remote_code=True).select(range(num_samples))
            options = ["no", "yes"] # 0 = not equivalent, 1 = equivalent
            
            def format_fn(ex):
                q = f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nAre these sentences equivalent?\nAnswer:"
                ans = ex["label"]
                return q, options, ans

        elif task == "Win": # WinoGrande (Coreference)
            dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True).select(range(num_samples))
            
            def format_fn(ex):
                # Winogrande requires filling a blank "_".
                # To use the existing prefix-masking logic, we set prefix to empty string
                # and score the WHOLE sentence with option 1 vs WHOLE sentence with option 2.
                # This ensures fluency is checked across the boundary.
                sent = ex["sentence"]
                opt1 = ex["option1"]
                opt2 = ex["option2"]
                
                # Replace _ with options
                full_opt1 = sent.replace("_", opt1)
                full_opt2 = sent.replace("_", opt2)
                
                q = "" # Empty prefix means we score the entire sequence
                opts = [full_opt1, full_opt2]
                ans = int(ex["answer"]) - 1 # Dataset uses "1" and "2", we need 0 and 1
                return q, opts, ans
        
        else:
            print(f"Skipping unknown task: {task}")
            continue

        # --- 2. Evaluation Loop ---
        correct = 0
        total = 0
        start_time = time.time()

        for example in tqdm(dataset, desc=f"Eval {task}"):
            # Format the data
            prefix_text, choices, answer_idx = format_fn(example)

            # Tokenize Prefix
            # Note: For Winogrande where prefix is "", length is 0 (or just BOS)
            prefix_ids = tokenizer(prefix_text, return_tensors="pt").input_ids
            prefix_len = prefix_ids.shape[1]

            # Prepare full sequences (Prefix + Choice)
            # If prefix is empty (Winogrande), just use choices
            if prefix_text:
                full_texts = [prefix_text + " " + choice for choice in choices]
            else:
                full_texts = choices

            # Tokenize Batch
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True)
            
            # Calculate Loss (Lower is better)
            losses = eval_choice_likelihood(model, tokenizer, device, inputs, prefix_len)

            predicted_idx = int(np.argmin(losses))
            
            if predicted_idx == answer_idx:
                correct += 1
            total += 1

        total_time = time.time() - start_time
        accuracy = correct / total
        speed = total / total_time
        results[task] = {"acc": accuracy, "speed": speed}
        
        print(f"[{task}] Result: Acc={accuracy:.2%} | Speed={speed:.2f} s/s")

    if is_system_mode:
        # Assuming print_moe_cache_stats is defined elsewhere in your scope
        try:
            print_moe_cache_stats(model)
        except NameError:
            pass
            
    return results


def moe_lightning_warmup(model, tokenizer, device_name, num_steps=10):
    print("[Warmup] Collecting stats...")
    dataset = load_dataset("tau/commonsense_qa", split="validation").select(range(num_steps))
    for ex in tqdm(dataset):
        text = f"Question: {ex['question']}\nAnswer: {ex['choices']['text'][0]}"
        inputs = tokenizer(text, return_tensors="pt").to(device_name)
        with torch.no_grad(): _ = model(**inputs)
   
if __name__ == "__main__":
    MODEL_ID = "mistralai/Mixtral-8x7B-v0.1" 
    
    # 2. Adjust Cache Size based on your GPU VRAM
    # 8x22B Experts are ~360MB per slot. Static model is ~15GB.
    # - For 24GB VRAM: Set to 16 (approx 6GB cache + 15GB model = 21GB)
    # - For 40GB VRAM: Set to 60 (approx 21GB cache + 15GB model = 36GB)
    # - For 80GB VRAM: Set to 128+
    MAX_GLOBAL_SLOTS = 32  
    
    NUM_SAMPLES = 50
    NUM_TTA_STEPS = 10
    set_seed(42)

    print("=========================================")
    print("🚀 System Run (OS-Style Quota + CLOCK)")
    print("=========================================")
    start = time.time()
    model, tokenizer = load_and_patch_system(MODEL_ID, max_global_slots=MAX_GLOBAL_SLOTS)
    
    # 더미 추론 (시스템 초기화)
    with torch.no_grad():
        _ = model(tokenizer("Hello", return_tensors="pt").to("cuda").input_ids)

    moe_lightning_warmup(model, tokenizer, "cuda", NUM_TTA_STEPS)
    results = evaluate_model(model, tokenizer, "cuda", True, num_samples=NUM_SAMPLES)
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("\n=========================================")
    print("🐢 Reference Run (CPU Only)")
    print("=========================================")
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu", torch_dtype=torch.float16)
    results = evaluate_model(ref_model, tokenizer, "cpu", False, num_samples=NUM_SAMPLES)

    # print(f"\n===================================================================")
    # print(f"🏆 Final Scoreboard")
    # print(f"System (GPU)   | {acc_sys:.2%}   | {speed_sys:.2f} s/s | {speed_sys/speed_cpu:.2f}x")
    # print(f"Reference (CPU)| {acc_cpu:.2%}   | {speed_cpu:.2f} s/s | 1.00x")