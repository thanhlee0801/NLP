import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import random
import joblib

# --- 1. Tạo dữ liệu giả (Dummy Data) ---
def generate_dummy_comments(num_comments=100):
    positive_comments = [
        "Sản phẩm tuyệt vời! Rất hài lòng.",
        "Dịch vụ khách hàng xuất sắc.",
        "Tôi yêu thích sản phẩm này, nó thật sự hữu ích.",
        "Trải nghiệm mua sắm tuyệt vời, sẽ quay lại.",
        "Chất lượng vượt trội so với giá tiền.",
        "Thật sự ấn tượng với hiệu suất của nó.",
        "Rất khuyến khích cho mọi người.",
        "Đây là thứ tôi cần, hoàn hảo!",
        "Chắc chắn sẽ mua lại lần nữa.",
        "Không thể tốt hơn được nữa."
    ]
    negative_comments = [
        "Sản phẩm tệ, lãng phí tiền bạc.",
        "Dịch vụ quá chậm và không chuyên nghiệp.",
        "Tôi thất vọng với chất lượng này.",
        "Không bao giờ mua lại từ cửa hàng này.",
        "Giá cao nhưng chất lượng kém.",
        "Thật đáng thất vọng, không như mong đợi.",
        "Không khuyên dùng cho bất kỳ ai.",
        "Hoàn toàn không hài lòng.",
        "Có nhiều lỗi và không hoạt động tốt.",
        "Đây là một sản phẩm tồi."
    ]
    neutral_comments = [
        "Sản phẩm đã được giao.",
        "Tôi đã nhận được hàng.",
        "Cần thêm thông tin về sản phẩm.",
        "Đánh giá về chức năng cơ bản.",
        "Sản phẩm hoạt động đúng như mô tả.",
        "Chỉ là một sản phẩm bình thường.",
        "Không có gì đặc biệt.",
        "Thông tin cần được xác minh.",
        "Đã sử dụng trong vài ngày.",
        "Không có bất kỳ cảm xúc nào."
    ]

    comments = []
    labels = []

    for _ in range(num_comments // 3):
        comments.append(random.choice(positive_comments))
        labels.append("tích cực")
        comments.append(random.choice(negative_comments))
        labels.append("tiêu cực")
        comments.append(random.choice(neutral_comments))
        labels.append("trung tính")

    # Đảm bảo đủ 100 bình luận nếu num_comments không chia hết cho 3
    while len(comments) < num_comments:
        comments.append(random.choice(positive_comments + negative_comments + neutral_comments))
        labels.append(random.choice(["tích cực", "tiêu cực", "trung tính"]))

    # Trộn ngẫu nhiên dữ liệu
    combined = list(zip(comments, labels))
    random.shuffle(combined)
    comments, labels = zip(*combined)

    return list(comments), list(labels)

comments, labels = generate_dummy_comments(num_comments=100)

print(f"Tổng số bình luận đã tạo: {len(comments)}")
print(f"Ví dụ bình luận và nhãn:\n{comments[0]} - {labels[0]}\n{comments[1]} - {labels[1]}")

# --- 2. Tiền xử lý dữ liệu ---

# Mã hóa nhãn cảm xúc thành số
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Chuyển nhãn số thành dạng one-hot encoding
one_hot_labels = tf.keras.utils.to_categorical(encoded_labels, num_classes=num_classes)

# Khởi tạo Tokenizer
# oov_token giúp xử lý các từ không có trong từ điển
tokenizer = Tokenizer(num_words=5000, oov_token="<unk>")
tokenizer.fit_on_texts(comments)

# Chuyển văn bản thành chuỗi số
sequences = tokenizer.texts_to_sequences(comments)

# Đệm chuỗi để có cùng độ dài
# max_len có thể được điều chỉnh tùy thuộc vào độ dài bình luận của bạn
max_len = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

print(f"\nKích thước từ điển: {len(tokenizer.word_index)}")
print(f"Độ dài chuỗi tối đa: {max_len}")
print(f"Hình dạng của dữ liệu đã đệm: {padded_sequences.shape}")
print(f"Hình dạng của nhãn one-hot: {one_hot_labels.shape}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, one_hot_labels, test_size=0.2, random_state=42)

print(f"\nKích thước tập huấn luyện: {X_train.shape[0]}")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]}")

# --- 3. Xây dựng mô hình CNN ---

embedding_dim = 100 # Kích thước vector nhúng
vocab_size = len(tokenizer.word_index) + 1 # Kích thước từ điển + 1 cho OOV token

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5), # Tăng cường khả năng tổng quát hóa
    Dense(num_classes, activation='softmax') # Đầu ra là số lớp cảm xúc
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Huấn luyện mô hình ---

epochs = 10 # Số epoch có thể được điều chỉnh
batch_size = 32 # Kích thước batch

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1, # Chia nhỏ tập huấn luyện để xác thực
                    verbose=1)

print("\nQuá trình huấn luyện hoàn tất.")

# --- 5. Đánh giá mô hình trên tập kiểm tra ---
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")

# --- 6. Dự đoán cảm xúc của các bình luận mới (ví dụ) ---
def predict_sentiment(text_list, model, tokenizer, max_len, label_encoder):
    sequences = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    predictions = model.predict(padded)
    # Lấy chỉ số của nhãn có xác suất cao nhất
    predicted_labels_encoded = np.argmax(predictions, axis=1)
    # Chuyển đổi lại về nhãn văn bản
    predicted_labels = label_encoder.inverse_transform(predicted_labels_encoded)
    return predicted_labels, predictions

print("\n--- Dự đoán cảm xúc cho các bình luận mới ---")
new_comments = [
    "Sản phẩm này thật tuyệt vời, tôi rất thích nó!",
    "Dịch vụ quá tệ, không đáng tiền chút nào.",
    "Bình luận này không có cảm xúc cụ thể.",
    "Tôi không chắc chắn về sản phẩm này.",
    "Khá tốt, nhưng có thể cải thiện thêm."
]

predicted_sentiments, probabilities = predict_sentiment(new_comments, model, tokenizer, max_len, label_encoder)

for i, comment in enumerate(new_comments):
    sentiment = predicted_sentiments[i]
    prob = probabilities[i]
    print(f"Bình luận: \"{comment}\"")
    print(f"Cảm xúc dự đoán: {sentiment}")
    print(f"Xác suất: {prob}")
    print(f"({label_encoder.classes_[0]}: {prob[0]:.2f}, {label_encoder.classes_[1]}: {prob[1]:.2f}, {label_encoder.classes_[2]}: {prob[2]:.2f})")
    print("-" * 30)
# --- Lưu mô hình và các đối tượng tiền xử lý ---
model.save('sentiment_cnn_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(max_len, 'max_len.pkl') # Lưu cả max_len để sử dụng khi pad_sequences

print("\nĐã lưu mô hình và các đối tượng tiền xử lý.")
