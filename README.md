<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/49e3f2ed-657a-4f5e-a007-2f06b6d1f4e9" />📌 Mô tả bài tập: Dự đoán tín dụng khách hàng với XGBoost
1. Bối cảnh

Bài toán xuất phát từ bộ dữ liệu Give Me Some Credit (cs-training.csv).
Mục tiêu: dự đoán khả năng khách hàng bị vỡ nợ (default) dựa trên thông tin tài chính và nhân khẩu học.

2. Các bước thực hiện
🔹 Chuẩn bị dữ liệu

Sử dụng file cs-training.csv.

Xử lý dữ liệu thiếu bằng SimpleImputer.

Chuẩn hóa dữ liệu bằng StandardScaler.

Chia dữ liệu thành tập huấn luyện và kiểm thử với train_test_split.

🔹 Huấn luyện mô hình

Mô hình sử dụng: XGBoost Classifier (XGBClassifier).

Tìm tham số tối ưu với RandomizedSearchCV và cross-validation (StratifiedKFold).

Dùng nhiều CPU để tăng tốc (multiprocessing).

🔹 Đánh giá mô hình

Sử dụng các thước đo:

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

ROC AUC Score

3. Kết quả mô hình

Hình bạn đưa là Confusion Matrix:

True Negative (TN) = 22,102 → dự đoán đúng khách hàng không vỡ nợ.

False Positive (FP) = 5,775 → dự đoán nhầm khách hàng không vỡ nợ thành có vỡ nợ.

False Negative (FN) = 424 → dự đoán nhầm khách hàng vỡ nợ thành không vỡ nợ.

True Positive (TP) = 1,578 → dự đoán đúng khách hàng vỡ nợ.

👉 Kết quả cho thấy mô hình nhận diện khá tốt khách hàng không vỡ nợ (TN nhiều), tuy nhiên còn bỏ sót một phần khách hàng vỡ nợ (FN = 424).

4. Ý nghĩa
Mô hình có thể dùng trong hệ thống quản lý rủi ro tín dụng của ngân hàng để hỗ trợ quyết định cho vay.

Việc giảm FN (424) rất quan trọng, vì đó là các khách hàng thực sự rủi ro nhưng bị mô hình bỏ sót
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/57f0240a-9143-4400-b195-d563a35921b4" />

Mô hình có thể dùng trong hệ thống quản lý rủi ro tín dụng của ngân hàng để hỗ trợ quyết định cho vay.

Việc giảm FN (424) rất quan trọng, vì đó là các khách hàng thực sự rủi ro nhưng bị mô hình bỏ sót.
