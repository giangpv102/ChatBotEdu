import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Lấy thư mục gốc của dự án (thư mục chứa app, rag, finetune)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn cài đặt
DATA_DIR = os.path.join(BASE_DIR, "rag", "data")
DB_DIR = os.path.join(BASE_DIR, "rag", "db")
# Mô hình embedding đa ngôn ngữ hỗ trợ tiếng Việt rất tốt
EMBEDDING_MODEL = "BAAI/bge-m3" 

def ingest_pdfs():
    print(f"Đang tìm kiếm file PDF trong thư mục {DATA_DIR}...")
    pdf_files = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print("Không tìm thấy file PDF nào. Vui lòng thêm file vào thư mục data.")
        return

    documents = []
    for pdf_path in pdf_files:
        print(f"Đang tải {pdf_path}...")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"Lỗi khi đọc file {pdf_path}: {e}")
            
    print(f"Tổng cộng có {len(documents)} trang tài liệu được tải.")
    
    # Chia nhỏ văn bản (chunking)
    print("Đang chia nhỏ văn bản (chunking)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn văn bản.")
    
    # Khởi tạo embedding model
    print(f"Đang tải mô hình embedding: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'}
    )
    
    # Tạo hoặc update Chroma DB
    print(f"Đang lưu vào Vector Database tại {DB_DIR}...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    
    print("Hoàn tất quá trình tạo Vector Database!")

if __name__ == "__main__":
    ingest_pdfs()
