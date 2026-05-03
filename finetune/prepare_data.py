import os
import json
import glob

def prepare_dataset(input_dir, output_file):
    """
    Đọc tất cả các file .jsonl trong input_dir,
    ghép reasoning_content và content của assistant thành dạng <think>...</think>\n...
    và lưu ra file output_file để dùng cho Unsloth / HuggingFace datasets.
    """
    all_data = []
    
    # Tìm tất cả file .jsonl trong các thư mục con
    search_pattern = os.path.join(input_dir, "**", "*.jsonl")
    file_list = glob.glob(search_pattern, recursive=True)
    
    print(f"Tìm thấy {len(file_list)} file .jsonl.")
    
    total_samples = 0
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    messages = data.get("messages", [])
                    new_messages = []
                    
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        
                        if role == "assistant" and "reasoning_content" in msg:
                            reasoning = msg["reasoning_content"]
                            # Ghép reasoning vào trong thẻ <think>
                            new_content = f"<think>\n{reasoning}\n</think>\n{content}"
                            new_messages.append({"role": role, "content": new_content})
                        else:
                            new_messages.append({"role": role, "content": content})
                    
                    if new_messages:
                        all_data.append({"messages": new_messages})
                        total_samples += 1
                        
                except json.JSONDecodeError:
                    print(f"Lỗi đọc JSON ở file {file_path}")
                    continue

    # Ghi ra file output
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in all_data:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Đã xử lý xong. Tổng cộng {total_samples} mẫu được lưu tại {output_file}.")

if __name__ == "__main__":
    # Đường dẫn thư mục chứa dữ liệu
    INPUT_DIR = "/home/giangpv102/Project/Chat_2/Vietnamese-Education-Reasoning-Chat"
    OUTPUT_FILE = "/home/giangpv102/Project/Chat_2/finetune/prepared_dataset.jsonl"
    
    prepare_dataset(INPUT_DIR, OUTPUT_FILE)
