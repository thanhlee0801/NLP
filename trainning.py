import numpy as np
import random
import joblib # Để lưu/tải các đối tượng Python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # Thay thế Tokenizer và pad_sequences
from sklearn.naive_bayes import MultinomialNB # Mô hình Naive Bayes
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report # Để đánh giá

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
        "Cần thêm thêm thông tin về sản phẩm.",
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

# Mã hóa nhãn cảm xúc thành số (Naive Bayes không cần one-hot encoding trực tiếp)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print(f"\nCác nhãn được mã hóa: {label_encoder.classes_}")

# Sử dụng TfidfVectorizer để chuyển đổi văn bản thành vector TF-IDF
# Đây là bước quan trọng thay thế cho Tokenizer và pad_sequences của CNN
vectorizer = TfidfVectorizer(max_features=5000) # Giới hạn số lượng đặc trưng
X_features = vectorizer.fit_transform(comments)

print(f"\nSố lượng đặc trưng (từ vựng): {len(vectorizer.vocabulary_)}")
print(f"Hình dạng của dữ liệu đã được vector hóa: {X_features.shape}")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_features, encoded_labels, test_size=0.2, random_state=42)

print(f"\nKích thước tập huấn luyện: {X_train.shape[0]}")
print(f"Kích thước tập kiểm tra: {X_test.shape[0]}")

# --- 3. Xây dựng và huấn luyện mô hình Naive Bayes ---

# Khởi tạo mô hình Multinomial Naive Bayes
model = MultinomialNB()

# Huấn luyện mô hình
model.fit(X_train, y_train)

print("\nQuá trình huấn luyện Naive Bayes hoàn tất.")

# --- 4. Đánh giá mô hình trên tập kiểm tra ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")

print("\nBáo cáo phân loại:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# --- 5. Lưu mô hình và các đối tượng tiền xử lý ---
try:
    # Lưu mô hình Naive Bayes
    joblib.dump(model, 'sentiment_naive_bayes_model.pkl')
    print("Đã lưu mô hình: sentiment_naive_bayes_model.pkl")

    # Lưu TfidfVectorizer
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Đã lưu TfidfVectorizer: tfidf_vectorizer.pkl")

    # Lưu LabelEncoder
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Đã lưu label encoder: label_encoder.pkl")

    print("\nQuá trình lưu file hoàn tất thành công.")
except Exception as e:
    print(f"Có lỗi xảy ra khi lưu file: {e}")

# --- 6. Dự đoán cảm xúc của các bình luận mới (ví dụ) ---
def predict_sentiment_nb(text_list, model, vectorizer, label_encoder):
    # Chuyển đổi văn bản mới sang vector TF-IDF bằng vectorizer đã huấn luyện
    text_features = vectorizer.transform(text_list)
    predictions = model.predict(text_features)
    probabilities = model.predict_proba(text_features) # Lấy xác suất

    # Chuyển đổi nhãn số thành nhãn văn bản
    predicted_labels = label_encoder.inverse_transform(predictions)

    results = []
    for i, text in enumerate(text_list):
        prob_dict = {label_encoder.classes_[j]: probabilities[i][j] for j in range(len(label_encoder.classes_))}
        results.append({
            "comment": text,
            "sentiment": predicted_labels[i],
            "probabilities": prob_dict
        })
    return results

print("\n--- Dự đoán cảm xúc cho các bình luận mới (Naive Bayes) ---")
new_comments = [
    "Sản phẩm này thật tuyệt vời, tôi rất thích nó!",
    "Dịch vụ quá tệ, không đáng tiền chút nào.",
    "Bình luận này không có cảm xúc cụ thể.",
    "Tôi không chắc chắn về sản phẩm này.",
    "Khá tốt, nhưng có thể cải thiện thêm.",
    "Tôi đã đặt hàng nhưng chưa nhận được, rất thất vọng.",
    "Chất lượng ổn, phù hợp với giá tiền."
]

predictions_nb = predict_sentiment_nb(new_comments, model, vectorizer, label_encoder)

for result in predictions_nb:
    print(f"Bình luận: \"{result['comment']}\"")
    print(f"Cảm xúc dự đoán: {result['sentiment']}")
    print("Xác suất:")
    for label, prob in result['probabilities'].items():
        print(f"  {label}: {prob:.2f}")
    print("-" * 30)
