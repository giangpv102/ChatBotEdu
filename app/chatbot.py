import gradio as gr
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
ADAPTER_PATH = os.path.join(BASE_DIR, "finetune", "lora_model") 
DB_DIR = os.path.join(BASE_DIR, "rag", "db")
EMBEDDING_MODEL = "BAAI/bge-m3"

print("Đang khởi tạo RAG Retriever...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    rag_enabled = True
    print("Khởi tạo RAG thành công!")
except Exception as e:
    print(f"Lỗi khởi tạo RAG (Có thể chưa chạy ingest.py): {e}")
    rag_enabled = False

print("Đang tải LLM...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto", 
    torch_dtype=torch.float16 if not torch.cuda.is_bf16_supported() else torch.bfloat16
)
# Tải weights từ quá trình fine-tune
try:
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("Tải adapter thành công!")
except Exception as e:
    print(f"Không tìm thấy adapter tại {ADAPTER_PATH}, đang chạy với base model.")

# --- 3. HÀM XỬ LÝ CHAT ---
def bot_stream(message, history):
    # 1. Lấy thông tin từ RAG nếu có
    context = ""
    if rag_enabled:
        docs = retriever.invoke(message)
        if docs:
            context = "Dựa vào các tài liệu sau đây để trả lời (nếu phù hợp):\n" + "\n---\n".join([d.page_content for d in docs])
    
    # 2. Xây dựng prompt
    system_prompt = "Bạn là giáo viên AI hỗ trợ học sinh Việt Nam từ Tiểu học đến THPT. Hãy trả lời câu hỏi học tập một cách chính xác, thân thiện, và dễ hiểu. Thường xuyên động viên học sinh và xưng hô là Thầy/Cô."
    
    # Chuyển đổi history sang format của ChatML
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, assistant_msg in history:
        # Loại bỏ khối <details> (suy luận) ra khỏi lịch sử để tiết kiệm token và tránh model bị "ảo giác"
        clean_assistant_msg = re.sub(r"<details.*?</details>", "", assistant_msg, flags=re.DOTALL).strip()
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": clean_assistant_msg})
    
    # Tin nhắn hiện tại của User (kèm context RAG)
    user_content = message
    if context:
        user_content = f"{context}\n\nCâu hỏi của học sinh: {message}"
        
    messages.append({"role": "user", "content": user_content})
    
    # 3. Chuẩn bị đầu vào cho model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 4. Cài đặt Streamer
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=2048,
        do_sample=True,  # BẮT BUỘC có khi dùng temperature
        temperature=0.3, # Giữ thấp để tránh sinh ngẫu nhiên kiến thức
        top_p=0.9,
    )
    
    # Bắt đầu thread sinh văn bản (Có try-except để bắt lỗi)
    def generate_with_catch():
        try:
            model.generate(**generation_kwargs)
        except Exception as e:
            print(f"\n[LỖI GENERATE]: {e}")
            
    thread = Thread(target=generate_with_catch)
    thread.start()
    
    # 5. Yield kết quả ra Gradio
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        
        # Xử lý hiển thị thẻ <think> trên giao diện Gradio
        display_text = partial_text
        if "<think>" in display_text and "</think>" not in display_text:
            # Đang trong quá trình suy nghĩ -> Mở block details
            display_text = display_text.replace("<think>", "<details open>\n<summary>🧠 AI đang suy luận...</summary>\n\n")
            display_text += "\n</details>" # Đóng tạm để Gradio không bị lỗi render HTML
        else:
            # Đã suy nghĩ xong -> Đóng block details lại cho gọn
            display_text = display_text.replace("<think>", "<details>\n<summary>🧠 Quá trình suy luận của giáo viên AI</summary>\n\n")
            display_text = display_text.replace("</think>", "\n</details>\n")
            
        yield display_text

# --- 4. GIAO DIỆN GRADIO ---
custom_css = """
#chatbot { height: 70vh !important; }
"""

with gr.Blocks(css=custom_css, title="Chatbot Giáo dục AI") as demo:
    gr.Markdown("# 🎓 Trợ lý Giáo viên AI (Fine-tuned & RAG)")
    gr.Markdown("Hệ thống được huấn luyện trên dataset Giáo dục Việt Nam và tích hợp RAG để đọc tài liệu PDF.")
    
    chatbot = gr.ChatInterface(
        fn=bot_stream,
        chatbot=gr.Chatbot(elem_id="chatbot", show_copy_button=True),
        textbox=gr.Textbox(placeholder="Nhập câu hỏi của em vào đây...", container=False, scale=7),
        theme="soft",
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
