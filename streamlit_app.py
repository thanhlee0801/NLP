import streamlit as st
import numpy as np
import os

# --- 1. Tiêu đề ứng dụng ---
st.title("Ứng dụng Dự đoán Cảm xúc Bình luận")
st.write("Sử dụng mô hình Naive Bayes để phân loại cảm xúc (tích cực, tiêu cực, trung tính) của các bình luận.")

# --- 2. Tải mô hình và các đối tượng tiền xử lý ---
@st.cache_resource # Sử dụng st.cache_resource để chỉ tải mô hình một lần
def load_resources():
    try:
        model_path = 'sentiment_naive_bayes_model.pkl'
        vectorizer_path = 'tfidf_vectorizer.pkl' # Đổi tên biến từ tokenizer_path
        label_encoder_path = 'label_encoder.pkl'

        # Kiểm tra sự tồn tại của các file
        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
            st.error("Lỗi: Không tìm thấy một hoặc nhiều file mô hình/tiền xử lý. Vui lòng đảm bảo các file này nằm cùng thư mục với streamlit_app.py")
            st.stop() # Dừng ứng dụng nếu file không tồn tại

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path) # Tải TfidfVectorizer
        label_encoder = joblib.load(label_encoder_path)
        st.success("Đã tải thành công mô hình Naive Bayes và các đối tượng.")
        return model, vectorizer, label_encoder # Trả về vectorizer thay vì tokenizer và không có max_len
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc đối tượng: {e}")
        st.stop() # Dừng ứng dụng nếu có lỗi khi tải

# Gán các giá trị trả về từ load_resources
model, vectorizer, label_encoder = load_resources() # Cập nhật các biến nhận


# --- 3. Hàm dự đoán cảm xúc ---
# Chú ý: Thay đổi các tham số đầu vào của hàm
def predict_sentiment(text, model, vectorizer, label_encoder):
    # Chuyển đổi văn bản sang vector TF-IDF bằng vectorizer đã tải
    text_features = vectorizer.transform([text])
    predictions = model.predict(text_features)
    probabilities = model.predict_proba(text_features)

    predicted_sentiment = label_encoder.inverse_transform(predictions)[0]

    probabilities_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        probabilities_dict[label] = probabilities[0][i]

    return predicted_sentiment, probabilities_dict

# --- 4. Giao diện người dùng Streamlit ---
user_comment = st.text_area("Nhập bình luận của bạn vào đây:", height=150, placeholder="Ví dụ: Sản phẩm này thật tuyệt vời!")

if st.button("Dự đoán Cảm xúc"):
    if user_comment:
        st.spinner("Đang phân tích cảm xúc...")
        # CHÚ Ý ĐÂY LÀ DÒNG BỊ LỖI TRƯỚC ĐÂY - Cập nhật cách gọi hàm predict_sentiment
        sentiment, probabilities = predict_sentiment(user_comment, model, vectorizer, label_encoder)

        st.subheader("Kết quả Dự đoán:")
        st.write(f"**Bình luận:** \"{user_comment}\"")

        color_map = {
            "tích cực": "green",
            "tiêu cực": "red",
            "trung tính": "orange"
        }
        sentiment_color = color_map.get(sentiment, "black")
        st.markdown(f"**Cảm xúc dự đoán:** <span style='color:{sentiment_color}; font-weight:bold;'>{sentiment.upper()}</span>", unsafe_allow_html=True)

        st.write("**Xác suất cho từng cảm xúc:**")
        for label, prob in probabilities.items():
            st.write(f"- **{label.capitalize()}**: {prob*100:.2f}%")
    else:
        st.warning("Vui lòng nhập bình luận để dự đoán.")
