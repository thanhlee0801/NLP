import streamlit as st
import numpy as np

import joblib
import os # Import os để kiểm tra sự tồn tại của file

@st.cache_resource
def load_resources():
    try:
        model_path = 'sentiment_naive_bayes_model.pkl' # Đổi tên file
        vectorizer_path = 'tfidf_vectorizer.pkl'       # Đổi tên file
        label_encoder_path = 'label_encoder.pkl'

        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, label_encoder_path]):
            st.error("Lỗi: Không tìm thấy một hoặc nhiều file mô hình/tiền xử lý. Vui lòng đảm bảo các file này nằm cùng thư mục với streamlit_app.py")
            st.stop()

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path) # Tải vectorizer
        label_encoder = joblib.load(label_encoder_path)
        st.success("Đã tải thành công mô hình Naive Bayes và các đối tượng.")
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc đối tượng: {e}")
        st.stop()

model, vectorizer, label_encoder = load_resources()

def predict_sentiment(text, model, vectorizer, label_encoder): # Thay đổi tham số
    # Chuyển đổi văn bản sang vector TF-IDF
    text_features = vectorizer.transform([text])
    predictions = model.predict(text_features)
    probabilities = model.predict_proba(text_features)

    predicted_sentiment = label_encoder.inverse_transform(predictions)[0]

    probabilities_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        probabilities_dict[label] = probabilities[0][i]

    return predicted_sentiment, probabilities_dict
    
# --- 1. Tiêu đề ứng dụng ---
st.title("Ứng dụng Dự đoán Cảm xúc Bình luận")
st.write("Sử dụng mô hình CNN để phân loại cảm xúc (tích cực, tiêu cực, trung tính) của các bình luận.")

# --- 2. Tải mô hình và các đối tượng tiền xử lý ---
@st.cache_resource # Sử dụng st.cache_resource để chỉ tải mô hình một lần
def load_resources():
    try:
        model_path = 'sentiment_cnn_model.h5'
        tokenizer_path = 'tokenizer.pkl'
        label_encoder_path = 'label_encoder.pkl'
        max_len_path = 'max_len.pkl'

        # Kiểm tra sự tồn tại của các file
        if not all(os.path.exists(p) for p in [model_path, tokenizer_path, label_encoder_path, max_len_path]):
            st.error("Lỗi: Không tìm thấy một hoặc nhiều file mô hình/tiền xử lý. Vui lòng đảm bảo các file này nằm cùng thư mục với streamlit_app.py")
            st.stop() # Dừng ứng dụng nếu file không tồn tại

        model = load_model(model_path)
        tokenizer = joblib.load(tokenizer_path)
        label_encoder = joblib.load(label_encoder_path)
        max_len = joblib.load(max_len_path)
        st.success("Đã tải thành công mô hình và các đối tượng.")
        return model, tokenizer, label_encoder, max_len
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình hoặc đối tượng: {e}")
        st.stop() # Dừng ứng dụng nếu có lỗi khi tải

model, tokenizer, label_encoder, max_len = load_resources()

# --- 3. Hàm dự đoán cảm xúc ---
def predict_sentiment(text, model, tokenizer, max_len, label_encoder):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    predictions = model.predict(padded)

    predicted_labels_encoded = np.argmax(predictions, axis=1)
    predicted_sentiment = label_encoder.inverse_transform(predicted_labels_encoded)[0]

    probabilities_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        probabilities_dict[label] = predictions[0][i]

    return predicted_sentiment, probabilities_dict

# --- 4. Giao diện người dùng Streamlit ---
user_comment = st.text_area("Nhập bình luận của bạn vào đây:", height=150, placeholder="Ví dụ: Sản phẩm này thật tuyệt vời!")

if st.button("Dự đoán Cảm xúc"):
    if user_comment:
        st.spinner("Đang phân tích cảm xúc...")
        sentiment, probabilities = predict_sentiment(user_comment, model, tokenizer, max_len, label_encoder)

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
