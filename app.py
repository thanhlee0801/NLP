import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Tải mô hình và các đối tượng tiền xử lý ---
try:
    model = load_model('sentiment_cnn_model.h5')
    tokenizer = joblib.load('tokenizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    max_len = joblib.load('max_len.pkl')
    print("Đã tải thành công mô hình và các đối tượng.")
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc đối tượng: {e}")
    # Xử lý lỗi, có thể thoát hoặc hiển thị thông báo lỗi trên web
    exit()

# --- Hàm dự đoán cảm xúc (tương tự như trong script gốc) ---
def predict_sentiment_for_web(text, model, tokenizer, max_len, label_encoder):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    predictions = model.predict(padded)

    # Lấy chỉ số của nhãn có xác suất cao nhất
    predicted_labels_encoded = np.argmax(predictions, axis=1)
    # Chuyển đổi lại về nhãn văn bản
    predicted_sentiment = label_encoder.inverse_transform(predicted_labels_encoded)[0]

    # Chuẩn bị xác suất cho hiển thị
    probabilities_dict = {}
    for i, label in enumerate(label_encoder.classes_):
        probabilities_dict[label] = predictions[0][i]

    return predicted_sentiment, probabilities_dict

# --- Định nghĩa các route cho ứng dụng Flask ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        comment = request.form['comment']
        if not comment:
            return render_template('index.html', sentiment="Vui lòng nhập bình luận để dự đoán.", comment="")

        predicted_sentiment, probabilities = predict_sentiment_for_web(comment, model, tokenizer, max_len, label_encoder)

        return render_template('index.html',
                               comment=comment,
                               sentiment=predicted_sentiment,
                               probabilities=probabilities)

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    # Đảm bảo file app.py và thư mục templates cùng cấp
    # Đảm bảo các file .h5, .pkl đã được tạo ra trong cùng thư mục
    app.run(debug=True) # debug=True giúp tự động tải lại khi có thay đổi code
