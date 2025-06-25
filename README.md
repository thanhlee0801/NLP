# NLP
1. Giới thiệu
Dự án này tập trung vào việc xây dựng một mô hình phân loại cảm xúc (Sentiment Analysis) sử dụng thuật toán Naive Bayes để phân loại các bình luận văn bản. Mục tiêu là gán nhãn cảm xúc cho các bình luận thành ba loại: tích cực, tiêu cực, và trung tính. Để minh họa và phát triển, chúng tôi đã sử dụng một tập dữ liệu nhỏ gồm 100 bình luận giả định.
________________________________________
2. Dữ liệu
•	Nguồn dữ liệu: 100 bình luận giả định được tạo ngẫu nhiên, mô phỏng các câu phát biểu có cảm xúc tích cực, tiêu cực và trung tính.
•	Các loại nhãn: Mỗi bình luận được gán một trong ba nhãn cảm xúc: "tích cực", "tiêu cực", hoặc "trung tính".
•	Chia tách dữ liệu: Tập dữ liệu được chia thành hai phần: 80% được sử dụng để huấn luyện mô hình (tập huấn luyện) và 20% còn lại được dùng để đánh giá hiệu suất của mô hình trên dữ liệu chưa từng thấy (tập kiểm tra).
________________________________________
3. Phương pháp tiếp cận: Naive Bayes
Chúng tôi đã triển khai mô hình Multinomial Naive Bayes, một biến thể đặc biệt phù hợp cho các tác vụ phân loại văn bản.
3.1. Cơ sở lý thuyết của Naive Bayes
Naive Bayes là một thuật toán phân loại xác suất dựa trên định lý Bayes. "Naive" (ngây thơ) ám chỉ giả định chính của nó: sự độc lập có điều kiện giữa các đặc trưng (từ) khi biết nhãn lớp. Mặc dù giả định này hiếm khi đúng hoàn toàn trong ngôn ngữ tự nhiên, Naive Bayes vẫn hoạt động hiệu quả đáng ngạc nhiên trong nhiều tác vụ phân loại văn bản do tính đơn giản và hiệu quả tính toán của nó.
Trong phân loại văn bản, mô hình học cách tính xác suất của một từ xuất hiện trong một bình luận thuộc một lớp cảm xúc cụ thể. Sau đó, khi một bình luận mới đến, nó sẽ tính xác suất bình luận đó thuộc về mỗi lớp cảm xúc dựa trên các từ chứa trong đó, và gán bình luận đó cho lớp có xác suất cao nhất.
3.2. Tiền xử lý dữ liệu cho Naive Bayes
Để chuẩn bị văn bản cho mô hình Naive Bayes, chúng tôi đã thực hiện các bước sau:
•	Mã hóa nhãn: Các nhãn cảm xúc dạng văn bản ("tích cực", "tiêu cực", "trung tính") được chuyển đổi thành các giá trị số nguyên (0, 1, 2) bằng cách sử dụng LabelEncoder của scikit-learn. Điều này là cần thiết vì các thuật toán học máy làm việc với dữ liệu số.
•	Vector hóa TF-IDF: Đây là bước chuyển đổi quan trọng nhất của văn bản. Chúng tôi sử dụng TfidfVectorizer từ scikit-learn để biến đổi các bình luận thô thành các vector số. 
o	TF (Term Frequency): Đo lường tần suất xuất hiện của một từ trong một bình luận cụ thể.
o	IDF (Inverse Document Frequency): Đo lường mức độ hiếm của một từ trong toàn bộ tập dữ liệu. Các từ xuất hiện thường xuyên trong nhiều bình luận (ví dụ: "và", "là") sẽ có IDF thấp, trong khi các từ đặc trưng hơn (ví dụ: "tuyệt vời", "thất vọng") sẽ có IDF cao.
o	Sự kết hợp của TF và IDF tạo ra một giá trị thể hiện tầm quan trọng của một từ trong một tài liệu cụ thể trong toàn bộ corpus. Vector hóa TF-IDF giúp giảm trọng số của các từ thông dụng và tăng trọng số của các từ mang ý nghĩa đặc trưng cho cảm xúc.
3.3. Xây dựng và Huấn luyện mô hình
•	Khởi tạo mô hình: Chúng tôi sử dụng lớp MultinomialNB() từ thư viện scikit-learn. Multinomial Naive Bayes đặc biệt phù hợp với các đặc trưng là số đếm (như tần suất từ hoặc trọng số TF-IDF).
•	Huấn luyện: Mô hình được huấn luyện bằng cách cung cấp ma trận TF-IDF của tập huấn luyện (X_train) và các nhãn số tương ứng (y_train) cho phương thức fit(). Quá trình này rất nhanh chóng.
________________________________________
4. Kết quả và Đánh giá
Sau khi huấn luyện, mô hình được đánh giá trên tập kiểm tra để đo lường hiệu suất của nó trên dữ liệu chưa từng thấy.
•	Độ chính xác (Accuracy): Đo lường tỷ lệ phần trăm các dự đoán đúng của mô hình.
•	Báo cáo phân loại (Classification Report): Cung cấp các số liệu chi tiết hơn cho từng lớp cảm xúc: 
o	Precision: Tỷ lệ các dự đoán tích cực của mô hình thực sự là tích cực.
o	Recall: Tỷ lệ các bình luận tích cực thực tế được mô hình dự đoán đúng là tích cực.
o	F1-score: Điểm trung bình điều hòa của Precision và Recall, cung cấp một cái nhìn cân bằng về hiệu suất.
o	Support: Số lượng mẫu thực tế trong mỗi lớp trong tập kiểm tra.
(Do tính chất giả định và quy mô nhỏ của tập dữ liệu (100 bình luận), các chỉ số hiệu suất cụ thể có thể không phản ánh chính xác hiệu quả của Naive Bayes trên các tập dữ liệu thực tế và lớn hơn. Tuy nhiên, chúng giúp minh họa quy trình.)
Ưu điểm của Naive Bayes trong dự án này:
•	Đơn giản và dễ hiểu: Cơ chế hoạt động dựa trên xác suất rõ ràng.
•	Hiệu quả tính toán: Cực kỳ nhanh trong cả quá trình huấn luyện và dự đoán, lý tưởng cho các tập dữ liệu lớn hoặc ứng dụng yêu cầu tốc độ.
•	Hiệu suất tốt với dữ liệu nhỏ: Thường mang lại kết quả hợp lý ngay cả với các tập dữ liệu huấn luyện khiêm tốn.
Hạn chế của Naive Bayes trong dự án này (và nói chung):
•	Giả định độc lập: Giả định mạnh mẽ rằng các từ độc lập với nhau, điều này hiếm khi đúng trong ngôn ngữ tự nhiên (ví dụ: "không tốt" khác với "tốt"). Điều này có thể hạn chế khả năng nắm bắt các mối quan hệ ngữ nghĩa phức tạp.
•	Kém hiệu quả với ngữ cảnh: Naive Bayes ít khả năng nắm bắt ngữ cảnh hoặc thứ tự từ so với các mô hình học sâu.
________________________________________
5. Triển khai và Lưu trữ Mô hình
Để ứng dụng có thể được sử dụng trong môi trường thực tế (như một ứng dụng web), mô hình đã huấn luyện và các đối tượng tiền xử lý cần được lưu lại.
•	Lưu trữ mô hình: Mô hình Naive Bayes đã huấn luyện (MultinomialNB) được lưu dưới dạng file sentiment_naive_bayes_model.pkl bằng thư viện joblib.
•	Lưu trữ Vectorizer: Đối tượng TfidfVectorizer cũng được lưu vào file tfidf_vectorizer.pkl. Điều này cực kỳ quan trọng vì khi dự đoán các bình luận mới, chúng phải được chuyển đổi thành vector TF-IDF bằng chính xác vectorizer đã được huấn luyện để đảm bảo tính nhất quán về từ vựng và trọng số.
•	Lưu trữ Label Encoder: Đối tượng LabelEncoder được lưu vào file label_encoder.pkl để có thể chuyển đổi kết quả dự đoán số học trở lại thành các nhãn cảm xúc dạng văn bản.
Các file .pkl này có thể dễ dàng được tải lại trong các ứng dụng khác (ví dụ: một ứng dụng Streamlit hoặc Flask) để thực hiện dự đoán mà không cần phải huấn luyện lại mô hình hoặc vectorizer.
________________________________________
6. Kết luận và Hướng phát triển
Dự án đã thành công trong việc minh họa một giải pháp phân loại cảm xúc hiệu quả bằng cách sử dụng mô hình Naive Bayes. Với tính đơn giản và hiệu quả, Naive Bayes là một điểm khởi đầu tuyệt vời cho các tác vụ phân loại văn bản, đặc biệt khi tài nguyên tính toán hạn chế hoặc cần tốc độ cao.
Để tiếp tục phát triển và cải thiện dự án này, các hướng sau có thể được xem xét:
•	Sử dụng tập dữ liệu thực tế lớn hơn: Đây là yếu tố quan trọng nhất để cải thiện độ chính xác và khả năng tổng quát hóa của mô hình trong thế giới thực.
•	Tiền xử lý văn bản nâng cao: Thực hiện các bước như loại bỏ stop words, chuẩn hóa văn bản (ví dụ: chuyển về chữ thường, xử lý dấu câu), và sử dụng lemmatization hoặc stemming để giảm nhiễu và chuẩn hóa từ vựng.
•	Thử nghiệm các biến thể Naive Bayes: Khám phá các mô hình Naive Bayes khác như Gaussian Naive Bayes (nếu đặc trưng là liên tục) hoặc Bernoulli Naive Bayes.
•	Kỹ thuật Feature Engineering: Tạo ra các đặc trưng bổ sung ngoài TF-IDF, chẳng hạn như độ dài bình luận, sự hiện diện của dấu chấm than, hoặc các từ cảm xúc mạnh (emoticons).
•	So sánh với các mô hình khác: Mặc dù báo cáo này tập trung vào Naive Bayes, việc so sánh hiệu suất với các mô hình phức tạp hơn (như CNN hoặc mô hình dựa trên Transformer) trên cùng một tập dữ liệu thực tế sẽ cung cấp cái nhìn toàn diện hơn về lựa chọn mô hình.

