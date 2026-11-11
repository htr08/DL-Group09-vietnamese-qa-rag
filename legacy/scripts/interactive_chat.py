import os
import sys

# Thêm thư mục gốc của project vào PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.rag.pipeline import MarketingRAGEngine
from backend.rag.config import rag_settings

def main():
    print("===========================================")
    print("🤖 CellphoneS AI Sales Assistant (CLI) 🤖")
    print("===========================================")
    
    print("\nĐang khởi tạo RAG Engine (kết nối Qdrant và Gemini)...")
    try:
        if not rag_settings.GEMINI_API_KEY:
            print("❌ Lỗi: Không tìm thấy GEMINI_API_KEY.")
            print("Vui lòng đảm bảo bạn đã cấu hình biến môi trường hoặc file .env")
            return
            
        engine = MarketingRAGEngine()
        print("✅ Khởi tạo thành công!\n")
        print("Hướng dẫn:")
        print("- Nhập câu hỏi của bạn và nhấn Enter.")
        print("- Gõ 'exit' hoặc 'quit' để thoát.")
        print("-------------------------------------------\n")
        
        while True:
            try:
                question = input("🧑 Bạn: ")
                if question.lower().strip() in ['exit', 'quit']:
                    print("🤖 Bot: Tạm biệt! Hẹn gặp lại bạn.")
                    break
                    
                if not question.strip():
                    continue
                    
                print("🤖 Bot: ", end="", flush=True)
                
                sources_found = []
                # Đọc stream kết quả từ RAG Engine
                for event in engine.ask_bot_stream(question):
                    if "chunk" in event:
                        print(event["chunk"], end="", flush=True)
                    if event.get("DONE"):
                        sources_found = event.get("sources", [])
                        
                print("\n")
                
                # In thông tin nguồn (sources) nếu có
                if sources_found:
                    print(f"\n   [ℹ️ Trích xuất từ {len(sources_found)} nguồn dữ liệu]")
                    for idx, src in enumerate(sources_found, 1):
                        filename = src.get('filename', 'Unknown')
                        score = src.get('score', 0.0)
                        print(f"   {idx}. {filename} (Độ tin cậy: {score:.2f})")
                print("\n-------------------------------------------")
                
            except KeyboardInterrupt:
                print("\n🤖 Bot: Tạm biệt!")
                break
            except Exception as e:
                print(f"\n❌ Đã xảy ra lỗi khi xử lý câu hỏi: {e}")
                
    except Exception as e:
        print(f"❌ Lỗi hệ thống: {e}")

if __name__ == "__main__":
    main()
