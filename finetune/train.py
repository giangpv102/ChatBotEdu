"""
Script Fine-tuning LLM bằng Unsloth trên Colab (đề xuất A100).
Bạn có thể copy nội dung file này vào Google Colab để chạy.
"""

import os
from unsloth import FastLanguageModel
from unsloth import get_chat_template
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Cấu hình mô hình
max_seq_length = 2048 # Có thể tăng lên 4096 hoặc 8192 trên A100 nếu bộ dữ liệu dài
dtype = None # Tự động chọn (A100 hỗ trợ bfloat16 rất tốt)
load_in_4bit = True # Dùng 4-bit quantization để tiết kiệm VRAM, kể cả trên A100 cũng giúp train batch lớn hơn

# Gợi ý: Qwen2.5-7B là model rất tốt cho Tiếng Việt và Reasoning. 
# Nếu A100 80GB, bạn có thể thử "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

print(f"Đang tải mô hình {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Thêm token HuggingFace nếu dùng model private
)

# 2. Cấu hình Chat Template
# Qwen2.5 dùng ChatML, ta dùng hàm của Unsloth để set chuẩn
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
    mapping = {"role" : "role", "content" : "content", "user" : "user", "assistant" : "assistant"}
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }
    
# 3. Thêm LoRA Adapters
print("Đang gắn LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Chọn 8, 16, 32, 64, 128 (16 là mức cân bằng tốt)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Tối ưu bằng 0
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Tải và xử lý dataset
# Lưu ý: Sửa lại đường dẫn file này khi bạn đưa lên Colab (ví dụ upload file prepared_dataset.jsonl lên Colab)
dataset_path = "/content/prepared_dataset.jsonl" if os.path.exists("/content/prepared_dataset.jsonl") else "prepared_dataset.jsonl"
print(f"Đang tải dataset từ {dataset_path}...")
dataset = load_dataset("json", data_files={"train": dataset_path}, split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Cấu hình Trainer
print("Khởi tạo Trainer...")
training_args = SFTConfig(
    per_device_train_batch_size = 4, # Trên A100 có thể tăng lên 8 hoặc 16
    gradient_accumulation_steps = 4,
    warmup_steps = 50,
    num_train_epochs = 2, 
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(), # A100 hỗ trợ bf16
    logging_steps = 10,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
    
    # [API MỚI TRL > 0.12] Các tham số này giờ phải nằm trong SFTConfig thay vì SFTTrainer
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
)

# [QUAN TRỌNG] Tắt hoàn toàn đa luồng (multiprocessing) của thư viện datasets
# để tránh triệt để lỗi PicklingError sinh ra do Unsloth và thư viện dill trên Python 3.12 của Colab.
import datasets
_original_map = datasets.Dataset.map
def _patched_map(self, *args, **kwargs):
    kwargs.pop("num_proc", None) # Loại bỏ num_proc để ép chạy đơn luồng (main thread)
    return _original_map(self, *args, **kwargs)
datasets.Dataset.map = _patched_map

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = training_args,
)

# 6. Bắt đầu Train
print("Bắt đầu huấn luyện...")
trainer_stats = trainer.train()

# 7. Lưu mô hình (Adapter)
output_model_path = "lora_model"
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print(f"Đã lưu adapter model tại thư mục {output_model_path}.")

# Ghi chú: Để merge ra GGUF sử dụng ollama/llama.cpp:
# model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
