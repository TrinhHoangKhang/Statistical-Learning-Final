import os

# PyTorch-only app: skip TensorFlow/Keras (avoids Keras 3 + Transformers conflict).
os.environ["USE_TF"] = "0"

# Streamlit's file watcher introspects torch.classes.__path__ and crashes on PyTorch 2.x.
import torch

torch.classes.__path__ = []

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- CẤU HÌNH GIAO DIỆN ---
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("Movie Review Sentiment Analyzer")
st.write("Ứng dụng sử dụng mô hình **DistilBERT** đã được fine-tune để dự đoán cảm xúc (Tích cực/Tiêu cực) của các bài đánh giá phim.")
st.markdown("---")

# --- CẤU HÌNH MÔ HÌNH HUGGINGFACE ---
HF_MODEL_ID = "TrinhHoangKhang/imdb-sentiment-with-distilbert" 

# --- TẢI MÔ HÌNH VÀO BỘ NHỚ ĐỆM (CACHE) ---
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        return_token_type_ids=False,
    )
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_id = int(probs.argmax(dim=-1).item())
    return {
        "label": model.config.id2label[pred_id],
        "score": float(probs[0, pred_id].item()),
    }

# Hiển thị trạng thái tải mô hình
with st.spinner("Đang tải mô hình từ Hugging Face Hub (Chỉ tải lần đầu)..."):
    try:
        tokenizer, model = load_model()
        st.success("Mô hình đã sẵn sàng hoạt động!")
    except Exception as e:
        st.error("Lỗi load mô hình! Bạn hãy kiểm tra lại biến `HF_MODEL_ID` xem đã đúng chưa.")
        st.info(f"Chi tiết lỗi: {e}")
        st.stop()

# --- GIAO DIỆN NHẬP DỮ LIỆU ---
st.subheader("Nhập bài Review phim của bạn:")
user_input = st.text_area(
    label="Hỗ trợ tiếng Anh (theo tập dữ liệu IMDB):",
    placeholder="Ví dụ: This movie was absolutely fantastic! The acting was brilliant...",
    height=150
)

# --- XỬ LÝ DỰ ĐOÁN ---
if st.button("Phân Tích Cảm Xúc", type="primary"):
    if not user_input.strip():
        st.warning("Vui lòng nhập một đoạn văn bản trước khi bấm phân tích!")
    else:
        with st.spinner("Mô hình đang suy luận..."):
            # Chạy dự đoán
            prediction = predict_sentiment(user_input, tokenizer, model)
            label = prediction['label']
            score = prediction['score']
            
            st.markdown("### Kết quả dự đoán:")
            
            # Kiểm tra nhãn trả về (LABEL_1/POSITIVE là Tích cực, LABEL_0/NEGATIVE là Tiêu cực)
            if label in ["LABEL_1", "POSITIVE"]:
                st.balloons() # Hiệu ứng bóng bay chúc mừng cho review tích cực
                st.success(f"### **Sentiment:** POSITIVE (Tích cực)")
            else:
                st.error(f"### **Sentiment:** NEGATIVE (Tiêu cực)")
                
            # Hiển thị độ tự tin dưới dạng phần trăm
            st.write(f"**Độ tự tin (Confidence Score):** `{score * 100:.2f}%`")