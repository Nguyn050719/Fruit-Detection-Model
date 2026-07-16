Hệ Thống Hỗ Trợ Nhận Diện Trái Cây (Fruit Classification)
    Dự án phân loại trái cây dựa trên các đặc điểm vật lý (khối lượng, chiều rộng, chiều cao, màu sắc) sử dụng thuật toán Học Máy (Machine Learning) kết hợp và giao diện web tương tác bằng Gradio.

Tổng quan về Mô hình
    Hệ thống sử dụng Ensemble Model (Voting Classifier), kết hợp sức mạnh của 3 thuật toán khác nhau nhằm tối ưu hóa độ chính xác:

        K-Nearest Neighbors (KNN): Chọn 17 láng giềng gần nhất (K=17).

        Decision Tree: Giới hạn độ sâu tối đa (max_depth) = 15, chia nhánh dựa trên tiêu chuẩn Entropy.

        Random Forest: Sử dụng 400 cây quyết định, chia nhánh dựa trên tiêu chuẩn Entropy.

        Voting Weights: Áp dụng tỷ lệ biểu quyết [1, 1, 3] (tăng trọng số cho Random Forest).

    Độ chính xác (Accuracy) trên tập kiểm thử đạt 87.17%.

Tính năng chính
    Dự đoán thời gian thực: Người dùng nhập các thông số (Khối lượng, Chiều cao, Chiều rộng, Nhóm màu) để AI dự đoán loại trái cây và đưa ra độ tin cậy.

    Cơ sở tri thức: Hệ thống tự động học các ngưỡng giá trị (min/max) từ dữ liệu gốc để đưa ra cảnh báo nếu thông số người dùng nhập vào là bất thường so với thực tế.

    Mẹo sử dụng: Đề xuất cách dùng, chế biến loại trái cây tương ứng (chỉ hiển thị khi độ chính xác >75%).

    Trực quan hóa: Sử dụng biểu đồ Radar để so sánh dữ liệu người dùng nhập vào với thông số tiêu chuẩn.

    Xử lý hàng loạt: Hỗ trợ tải lên file CSV để dự đoán tự động cho nhiều mẫu cùng lúc.

    Lưu lịch sử & Xuất Excel: Theo dõi kết quả dự đoán và xuất báo cáo ra file .xlsx kèm theo thống kê chi tiết.

Cấu trúc thư mục
    app.py: Mã nguồn chính chạy giao diện web Gradio.

    Mo_hinh_huan_luyen.ipynb: File Jupyter Notebook chứa quá trình huấn luyện và đánh giá mô hình.

    Mo_hinh_da_huan_luyen.joblib & Chuan_hoa_du_lieu.joblib: Các tệp lưu trữ mô hình và bộ chuẩn hóa dữ liệu.

    Du_lieu_mau.csv: Tập dữ liệu dùng để huấn luyện.

    requirements.txt: Danh sách các thư viện cần thiết để chạy dự án.

    image/: Thư mục chứa hình ảnh minh họa cho các loại trái cây.

Hướng dẫn cài đặt và sử dụng:
    Bước 1: Tạo môi trường ảo (venv)
        python -m venv venv

    Bước 2: Kích hoạt môi trường ảo
        - Trên Windows (PowerShell):
            venv\Scripts\Activate.ps1

        - Trên Windows (CMD/Command Prompt):
            venv\Scripts\activate

        - Trên macOS/Linux:
            source venv/bin/activate

    Bước 3: Cài đặt các thư viện cần thiết
        pip install -r requirements.txt

    Bước 4: Huấn luyện mô hình để tạo các tệp .joblib
        Vì các file mô hình dung lượng lớn (`Mo_hinh_da_huan_luyen.joblib` và `Chuan_hoa_du_lieu.joblib`) bị bỏ qua không đẩy lên GitHub, bạn cần chạy huấn luyện mô hình để tự sinh ra các file này trước:
                - Cách 1 (Khuyên dùng): Mở file `Mo_hinh_huan_luyen.ipynb` bằng VS Code hoặc Jupyter Notebook, chọn kernel là môi trường ảo `venv` vừa cài đặt, và nhấn **Run All** để huấn luyện mô hình.
                
                - Cách 2 (Sử dụng Terminal): Chạy lệnh sau để tự động huấn luyện thông qua Notebook ngay trên Command Line: jupyter nbconvert --to notebook --execute --inplace Mo_hinh_huan_luyen.ipynb
    Bước 5: Khởi chạy ứng dụng
        python app.py

    Hệ thống sẽ cung cấp một đường link cục bộ (thường là `http://127.0.0.1:7860/`). Hãy click vào đường link đó hoặc copy và dán vào trình duyệt web để bắt đầu sử dụng!