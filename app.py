import gradio as gr
import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import joblib
from math import pi
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sử dụng backend 'Agg' để chạy mượt trên web
matplotlib.use('Agg')

# ==========================================
# 1. CẤU HÌNH & LOAD MODEL
# ==========================================

DATA_PATH = 'Du_lieu_mau.csv'
MODEL_PATH = 'Mo_hinh_da_huan_luyen.joblib'
SCALER_PATH = 'Chuan_hoa_du_lieu.joblib'

# Biến toàn cục
algorithm_info = {
    "params": {
        "KNN": "17 láng giềng (K=17)",
        "Decision Tree": "Độ sâu max: 15, Tiêu chuẩn: Entropy",
        "Random Forest": "400 cây quyết định, Tiêu chuẩn: Entropy",
        "Ensemble Weights": "Random Forest (x3) - KNN (x1) - DT (x1)"
    },
    "accuracy": {
        "dataset_name": "Chưa xác định",
        "KNN": 0.0, "DT": 0.0, "RF": 0.0, "Ensemble": 0.0
    }
}

vn_names = {
    'apple': 'Táo', 'mandarin': 'Quýt', 'orange': 'Cam', 'lemon': 'Chanh vàng',
    'pear': 'Lê', 'peach': 'Đào', 'mango': 'Xoài', 'banana': 'Chuối',
    'pomegranate': 'Lựu', 'tomato': 'Cà chua',
    'strawberry': 'Dâu tây', 'grape': 'Nho', 'watermelon': 'Dưa hấu',
    'pineapple': 'Dứa/Thơm', 'kiwi': 'Kiwi', 'cherry': 'Cherry',
    'avocado': 'Bơ', 'papaya': 'Đu đủ', 'dragon_fruit': 'Thanh long',
    'guava': 'Ổi', 'plum': 'Mận', 'lime': 'Chanh ta',
    'grapefruit': 'Bưởi', 'durian': 'Sầu riêng', 'coconut': 'Dừa',
    'lychee': 'Vải', 'starfruit': 'Khế', 'persimmon': 'Hồng',
    'passion_fruit': 'Chanh dây', 'apricot': 'Mơ'
}

# --- MỚI: TỪ ĐIỂN GỢI Ý SỬ DỤNG ---
fruit_usage_info = {
    'apple': "🍎 **Táo:** Nên ăn cả vỏ (sau khi rửa sạch) để lấy vitamin. Làm nước ép hoặc bánh táo nướng rất ngon.",
    'mandarin': "🍊 **Quýt:** Dễ tiêu hóa, ăn trực tiếp. Vỏ quýt phơi khô làm vị thuốc trần bì.",
    'orange': "🍊 **Cam:** Vắt nước uống tăng đề kháng hoặc ăn múi. Vỏ cam làm mứt.",
    'lemon': "🍋 **Chanh vàng:** Làm bánh lemon tart, pha trà hoặc làm nước sốt salad.",
    'pear': "🍐 **Lê:** Tính mát, trị ho. Có thể ăn tươi hoặc hấp đường phèn.",
    'peach': "🍑 **Đào:** Làm trà đào, mứt đào. Ăn tươi giòn ngọt.",
    'mango': "🥭 **Xoài:** Xoài chín ăn tráng miệng, xoài xanh làm gỏi/nộm. Xôi xoài là đặc sản.",
    'banana': "🍌 **Chuối:** Ăn khi chín vàng (có đốm đen càng ngọt). Làm bánh chuối, chè chuối.",
    'pomegranate': "🔴 **Lựu:** Ép nước uống rất tốt cho tim mạch và da.",
    'tomato': "🍅 **Cà chua:** Nấu canh, làm sốt pasta hoặc ăn sống salad. Giàu Lycopene.",
    'strawberry': "🍓 **Dâu tây:** Ăn kèm sữa chua, làm sinh tố hoặc trang trí bánh kem.",
    'grape': "🍇 **Nho:** Rửa sạch phấn trắng ăn cả vỏ. Có thể sấy khô hoặc làm rượu.",
    'watermelon': "🍉 **Dưa hấu:** Giải nhiệt mùa hè. Vỏ dưa hấu có thể nấu canh.",
    'pineapple': "🍍 **Dứa:** Ăn tươi, xào thịt bò, nấu canh chua. Nước ép dứa hỗ trợ giảm cân.",
    'kiwi': "🥝 **Kiwi:** Gọt vỏ hoặc bổ đôi xúc thìa. Giàu Vitamin C gấp nhiều lần cam.",
    'cherry': "🍒 **Cherry:** Ăn tươi là ngon nhất. Rất tốt cho giấc ngủ.",
    'avocado': "🥑 **Bơ:** Làm sinh tố bơ, dầm đường hoặc ăn kèm bánh mì nướng trứng.",
    'papaya': "🟠 **Đu đủ:** Chín ăn nhuận tràng. Xanh làm nộm bò khô, hầm xương.",
    'dragon_fruit': "🐉 **Thanh long:** Tính mát, ít calo. Trộn salad trái cây rất đẹp.",
    'guava': "🍐 **Ổi:** Hàm lượng Vitamin C cực cao. Nên ăn cả vỏ, bỏ ruột nếu đau dạ dày.",
    'plum': "🟣 **Mận:** Ăn tươi chấm muối ớt, làm mứt mận hoặc siro.",
    'lime': "🍈 **Chanh ta:** Gia vị không thể thiếu cho phở, bún. Làm nước chanh muối.",
    'grapefruit': "🟠 **Bưởi:** Ăn tép bưởi giảm cân. Vỏ bưởi nấu chè, gội đầu.",
    'durian': "🟡 **Sầu riêng:** Ăn múi trực tiếp. Làm bánh pía, kem, chè sầu riêng.",
    'coconut': "🥥 **Dừa:** Uống nước, nạo cùi kho thịt. Nước cốt dừa nấu chè.",
    'lychee': "🔴 **Vải:** Ăn tươi, sấy khô (vải thiều). Làm trà vải giải nhiệt.",
    'starfruit': "⭐ **Khế:** Khế chua nấu canh chua cá lóc. Khế ngọt ăn tráng miệng.",
    'persimmon': "🟠 **Hồng:** Hồng giòn ăn tươi. Hồng mềm để lạnh ăn như kem.",
    'passion_fruit': "🟣 **Chanh dây:** Pha nước uống với đường/mật ong. Làm sốt cho món Âu.",
    'apricot': "🟠 **Mơ:** Thường dùng ngâm đường, làm ô mai hoặc ngâm rượu mơ."
}

# Khởi tạo biến
model = None
scaler = None
class_accuracy = {}
label_map = {}
label_eng_map = {} # MỚI: Map ID sang tên tiếng Anh
fruit_classes_list = []
unique_fruits = None
fruit_means = None
# --- BIẾN: Lưu kiến thức đã học ---
fruit_knowledge_base = {} 
# ---------------------------------
counts = None
fixed_accuracy_str = "0%"

# --- HÀM LOAD HỆ THỐNG ---
def load_system():
    global model, scaler, class_accuracy, label_map, label_eng_map, fruit_classes_list, unique_fruits, fruit_means, counts, fixed_accuracy_str, algorithm_info, fruit_knowledge_base
    
    # 1. Load Model & Scaler
    try:
        print("--> Đang tải Model & Scaler...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Đã load Model thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file .joblib.")
        return

    # 2. Đọc dữ liệu gốc
    try:
        df = pd.read_csv(DATA_PATH)
        feature_names = ['mass', 'width', 'height', 'color_score']
        
        X = df[feature_names]
        y = df['fruit_label']

        X_scaled = scaler.transform(X) 
        _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0) 
        
        knn_model = model.named_estimators_['knn']
        dt_model = model.named_estimators_['dt']
        rf_model = model.named_estimators_['rf']

        acc_knn = knn_model.score(X_test, y_test)
        acc_dt = dt_model.score(X_test, y_test)
        acc_rf = rf_model.score(X_test, y_test)
        acc_ens = model.score(X_test, y_test)
        
        algorithm_info["accuracy"] = {
            "dataset_name": DATA_PATH,
            "KNN": acc_knn, "DT": acc_dt, "RF": acc_rf, "Ensemble": acc_ens
        }
        
        fixed_accuracy_str = f"{acc_ens * 100:.2f}%"

        y_pred_test = model.predict(X_test)
        report = classification_report(y_test, y_pred_test, output_dict=True)
        
        class_accuracy = {}
        for label_str, metrics in report.items():
            if label_str.isdigit():
                class_accuracy[int(label_str)] = metrics['precision']

        fruit_means = df.groupby('fruit_label')[feature_names].mean()
        
        # --- LOGIC: TỰ ĐỘNG HỌC NGƯỠNG (CHÍNH XÁC TUYỆT ĐỐI) ---
        print("--> Đang học dữ liệu chi tiết từ file CSV...")
        stats = df.groupby('fruit_label')[feature_names].agg(['min', 'max'])
        fruit_knowledge_base = stats.to_dict('index')
        # ---------------------------------

        unique_fruits = df[['fruit_label', 'fruit_name']].drop_duplicates().sort_values('fruit_label')
        
        label_map = {}
        label_eng_map = {} # Reset
        fruit_classes_list = []
        
        for _, row in unique_fruits.iterrows():
            eng_name = row['fruit_name']
            vn_name = vn_names.get(eng_name, eng_name.capitalize())
            label_map[row['fruit_label']] = vn_name
            label_eng_map[row['fruit_label']] = eng_name # MỚI: Lưu tên tiếng Anh
            fruit_classes_list.append(vn_name)
        
        counts = df['fruit_name'].map(vn_names).value_counts().reset_index()
        counts.columns = ['Loại quả', 'Số lượng mẫu']

    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy {DATA_PATH}.")

load_system()

# ==========================================
# 2. UI & LOGIC
# ==========================================
def generate_algo_info_markdown():
    acc = algorithm_info["accuracy"]
    params = algorithm_info["params"]
    md = f"""
    # Thông tin mô hình


    
    ### Cấu hình chi tiết
    Model được load từ file: `{MODEL_PATH}`
    
    | Thuật toán | Tham số kỹ thuật | Giải thích ngắn |
    | :--- | :--- | :--- |
    | **K-Nearest Neighbors** | `{params['KNN']}` | Chọn 17 điểm dữ liệu gần nhất để bỏ phiếu. |
    | **Decision Tree** | `{params['Decision Tree']}` | Cây quyết định phân nhánh dựa trên độ hỗn loạn tin (Entropy). |
    | **Random Forest** | `{params['Random Forest']}` | Kết hợp 400 cây quyết định để giảm sai số. |
    
    ---
    
    ### Dữ liệu gốc
    File dữ liệu được dùng để huấn luyện model:
    """
    return md

# --- HÀM MỚI: Trả về file dữ liệu mẫu ---
def return_data_file():
    return DATA_PATH
# ----------------------------------------

image_dir = "image"
def get_image_path(fruit_label):
    if unique_fruits is None: return None
    try:
        eng_name = unique_fruits[unique_fruits['fruit_label'] == fruit_label]['fruit_name'].values[0]
        return os.path.join(image_dir, f"{eng_name}.jpg")
    except: return None

def core_prediction(mass, width, height, color_score):
    # Trả về thêm 1 biến suggestion_text (mặc định rỗng)
    if model is None: return "N/A", None, 0, 0, "Lỗi: Chưa load model", {}, "0s", ""
    start_time = time.time()
    
    X_new = [[mass, width, height, color_score]]
    X_new_scaled = scaler.transform(X_new)

    prediction_label = model.predict(X_new_scaled)[0]
    probabilities = model.predict_proba(X_new_scaled)[0]
    confidence_score = max(probabilities)
    precision_val = class_accuracy.get(prediction_label, 0.0)
    
    fruit_name_vn = label_map.get(prediction_label, "Unknown")
    
    prob_dict = {fruit_classes_list[i]: float(probabilities[i]) for i in range(len(fruit_classes_list))}

    # --- LOGIC KIỂM TRA SIÊU NGHIÊM NGẶT ---
    advice_msg = "..."
    warnings = []
    
    if prediction_label in fruit_knowledge_base:
        kbase = fruit_knowledge_base[prediction_label]
        
        # Kiểm tra thông số (Giữ nguyên logic cũ)
        min_mass, max_mass = kbase[('mass', 'min')], kbase[('mass', 'max')]
        if mass < min_mass or mass > max_mass:
            warnings.append(f"- Khối lượng {mass}g nằm ngoài vùng chuẩn ({int(min_mass)}-{int(max_mass)}g).")
            
        min_w, max_w = kbase[('width', 'min')], kbase[('width', 'max')]
        if width < min_w or width > max_w:
            warnings.append(f"- Chiều rộng {width}cm nằm ngoài vùng chuẩn ({min_w:.1f}-{max_w:.1f}cm).")

        min_h, max_h = kbase[('height', 'min')], kbase[('height', 'max')]
        if height < min_h or height > max_h:
            warnings.append(f"- Chiều cao {height}cm nằm ngoài vùng chuẩn ({min_h:.1f}-{max_h:.1f}cm).")
            
        min_c, max_c = kbase[('color_score', 'min')], kbase[('color_score', 'max')]
        if color_score < min_c or color_score > max_c:
             warnings.append(f"- Màu sắc {color_score} không khớp với dữ liệu ({min_c:.2f}-{max_c:.2f}).")

        # XỬ LÝ KẾT QUẢ ADVICE
        if warnings:
            precision_val = 0.0 # Ép độ chính xác về 0 nếu dữ liệu bất thường
            advice_msg = "Phát hiện thông số bất thường:\n" + "\n".join(warnings) + "\n\n-> Kết quả dự đoán bị bác bỏ do dữ liệu không khớp chuẩn."
        else:
            if precision_val >= 0.90: advice_msg = f"Dữ liệu hoàn toàn khớp với hồ sơ. Độ tin cậy cao."
            elif precision_val >= 0.75: advice_msg = f"Dữ liệu nằm trong vùng cho phép."
            else: advice_msg = f"Kết quả chỉ mang tính tham khảo."
    
    # --- MỚI: LOGIC GỢI Ý SỬ DỤNG ---
    suggestion_text = ""
    # Chỉ hiển thị khi Hiệu suất thực >= 75% và không có cảnh báo
    if precision_val >= 0.75 and not warnings:
        eng_name = label_eng_map.get(prediction_label, "")
        if eng_name in fruit_usage_info:
             suggestion_text = f"💡 **Mách bạn:** {fruit_usage_info[eng_name]}"
    # --------------------------------

    end_time = time.time()
    time_str = f"{(end_time - start_time):.4f}s"

    img_path = get_image_path(prediction_label)
    display_img = img_path if img_path and os.path.exists(img_path) else None

    # Biểu đồ Radar (Giữ nguyên)
    bg_color = '#111111' 
    text_color = '#EEEEEE'
    accent_color = '#FF8C00' 
    plt.style.use('dark_background') 
    fig = plt.figure(figsize=(12, 5)) 
    fig.patch.set_facecolor(bg_color)
    
    ax1 = fig.add_subplot(121, polar=True)
    ax1.set_facecolor(bg_color)
    categories = ['Khối lượng', 'Rộng', 'Cao', 'Màu']
    angles = [n / 4 * 2 * pi for n in range(4)]
    angles += angles[:1] 
    ax1.set_theta_offset(pi / 2)
    ax1.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color=text_color)
    plt.ylim(0, 1)
    ax1.grid(color='#444444', linestyle='--')
    ax1.spines['polar'].set_color('#444444')

    try:
        predicted_mean_scaled = scaler.transform([fruit_means.loc[prediction_label]])[0].tolist()
        predicted_mean_scaled += predicted_mean_scaled[:1]
        ax1.plot(angles, predicted_mean_scaled, linestyle='dashed', color='#00CCFF', label='Tiêu chuẩn')
    except: pass

    user_scaled_plot = X_new_scaled[0].tolist()
    user_scaled_plot += user_scaled_plot[:1]
    
    plot_color = '#FF0033' if len(warnings) > 0 else accent_color
    
    ax1.plot(angles, user_scaled_plot, linewidth=2, color=plot_color, label='Input')
    ax1.fill(angles, user_scaled_plot, color=plot_color, alpha=0.3)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor(bg_color)
    
    sorted_indices = np.argsort(probabilities)[::-1][:5]
    top_names = [fruit_classes_list[i] for i in sorted_indices]
    top_probs = [probabilities[i] for i in sorted_indices]
    
    bars = ax2.bar(top_names, top_probs, color=[plot_color if i==0 else '#555' for i in range(5)])
    for bar in bars:
        h = bar.get_height()
        if h>0.01: ax2.text(bar.get_x()+bar.get_width()/2, h, f'{h*100:.0f}%', ha='center', va='bottom', color='white')
    ax2.set_title("Top 5 dự đoán (Vote AI)", color='white')
    plt.xticks(rotation=15)
    plt.tight_layout()

    # Trả về thêm suggestion_text vào cuối
    return fruit_name_vn, display_img, confidence_score, precision_val, advice_msg, fig, prob_dict, time_str, suggestion_text

def calculate_stats_text(history):
    if not history: return "Chưa có dữ liệu"
    df_hist = pd.DataFrame(history, columns=["STT", "Mass", "Width", "Height", "Color", "Dự đoán", "Conf %", "Real Acc %"])
    avg_real_acc = df_hist["Real Acc %"].mean()
    total_samples = len(df_hist)
    return f"Tổng số mẫu: {total_samples} | Trung bình độ chính xác thực tế: {avg_real_acc:.2f}%"

def preview_prediction(mass, width, height, color_score):
    # Nhận thêm suggestion
    name, img, conf, prec, advice, fig, prob_dict, time_val, suggestion = core_prediction(mass, width, height, color_score)
    # Trả về thêm suggestion
    return prob_dict, img, f"{conf * 100:.2f}%", f"{prec * 100:.2f}%", advice, time_val, fig, suggestion

def save_prediction(mass, width, height, color_score, history):
    # Nhận thêm suggestion
    name, img, conf, prec, advice, fig, prob_dict, time_val, suggestion = core_prediction(mass, width, height, color_score)
    stt = len(history) + 1
    new_row = [stt, mass, width, height, color_score, name, conf * 100, prec * 100]
    history.append(new_row)
    history_df = pd.DataFrame(history, columns=["STT", "Mass", "Width", "Height", "Color", "Dự đoán", "Conf %", "Real Acc %"])
    # Trả về thêm suggestion (đúng vị trí trong outputs)
    return prob_dict, img, f"{conf * 100:.2f}%", f"{prec * 100:.2f}%", advice, time_val, fig, suggestion, history_df, calculate_stats_text(history), history

def process_upload(file_obj, history):
    if file_obj is None: 
        return history, pd.DataFrame(), "Vui lòng tải file lên!", "Chưa có dữ liệu", generate_algo_info_markdown(), return_data_file() # THÊM return_data_file()
    try:
        df_new = pd.read_csv(file_obj.name)
        cols = ['mass', 'width', 'height', 'color_score']
        
        if not all(c in df_new.columns for c in cols): 
            return history, pd.DataFrame(), f"Thiếu cột {cols}", calculate_stats_text(history), generate_algo_info_markdown(), return_data_file() # THÊM return_data_file()
        
        X_batch = scaler.transform(df_new[cols])
        preds = model.predict(X_batch)
        probs = model.predict_proba(X_batch)
        
        count = 0
        current_stt = len(history)
        for i in range(len(df_new)):
            current_stt += 1
            mass, width, height, color = df_new[cols].iloc[i]
            pred_label = preds[i]
            fruit_name = label_map.get(pred_label, "Unknown")
            conf = max(probs[i]) * 100
            real_acc = class_accuracy.get(pred_label, 0.0) * 100
            history.append([current_stt, mass, width, height, color, fruit_name, conf, real_acc])
            count += 1
            
        history_df = pd.DataFrame(history, columns=["STT", "Mass", "Width", "Height", "Color", "Dự đoán", "Conf %", "Real Acc %"])
        msg = f"Đã thêm {count} dòng!"

        new_info_md = generate_algo_info_markdown()
        if 'fruit_label' in df_new.columns:
            y_true_new = df_new['fruit_label']
            knn_sub = model.named_estimators_['knn']
            dt_sub = model.named_estimators_['dt']
            rf_sub = model.named_estimators_['rf']

            new_acc_knn = knn_sub.score(X_batch, y_true_new)
            new_acc_dt = dt_sub.score(X_batch, y_true_new)
            new_acc_rf = rf_sub.score(X_batch, y_true_new)
            new_acc_ens = model.score(X_batch, y_true_new)
            
            algorithm_info["accuracy"]["dataset_name"] = os.path.basename(file_obj.name)
            algorithm_info["accuracy"]["KNN"] = new_acc_knn
            algorithm_info["accuracy"]["DT"] = new_acc_dt
            algorithm_info["accuracy"]["RF"] = new_acc_rf
            algorithm_info["accuracy"]["Ensemble"] = new_acc_ens
            
            new_info_md = generate_algo_info_markdown()
            msg += f" (Đã cập nhật độ chính xác theo file)"

        return history, history_df, msg, calculate_stats_text(history), new_info_md, return_data_file() # THÊM return_data_file()

    except Exception as e:
        return history, pd.DataFrame(), f"Lỗi: {str(e)}", calculate_stats_text(history), generate_algo_info_markdown(), return_data_file() # THÊM return_data_file()

def export_to_excel(history):
    if not history: return None
    try:
        # Đảm bảo các cột tiếng Việt này khớp với thứ tự trong history
        columns = ["STT", "Khối lượng", "Chiều rộng", "Chiều cao", "Màu sắc", "Dự đoán", "Độ tin cậy", "Độ chính xác"]
        df_hist = pd.DataFrame(history, columns=columns)
        
        # 1. KHẮC PHỤC LỖI: Sử dụng tên cột "Độ chính xác" (sau khi đổi từ "Real Acc %")
        avg_real_acc = df_hist["Độ chính xác"].mean() # Đã sửa
        
        fruit_counts = df_hist["Dự đoán"].value_counts().reset_index()
        fruit_counts.columns = ["Loại Quả", "Số lượng"]
        
        file_path = "ket_qua_phan_loai.xlsx"
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            df_hist.to_excel(writer, sheet_name="Lịch sử chi tiết", index=False)
            
            # 2. SỬA CÁCH TẠO summary_df cho phù hợp với dữ liệu đã tính
            summary_df = pd.DataFrame([
                ["Tổng số mẫu", len(df_hist)], 
                ["TB độ chính xác thực tế", f"{avg_real_acc:.2f}%"]
            ], columns=["Thống kê", "Giá trị"]) # Đã sửa
            
            summary_df.to_excel(writer, sheet_name="Thống kê", index=False)
            fruit_counts.to_excel(writer, sheet_name="Thống kê", index=False, startrow=4)
        return file_path
    except Exception as e:
        print(f"Lỗi xuất Excel: {e}")
        return None
    
# --- CSS ---
custom_css = """
footer {visibility: hidden !important}
.tab-nav button {
    font-weight: 900 !important;
    font-size: 1.1em !important;
}
#output_image {
    height: 320px !important;
    display: flex !important;
    justify_content: center !important;
    align-items: center !important;
    overflow: hidden !important;
}
#output_image div:has(img),
#output_image button:has(img),
#output_image > div,
#output_image > div > button,
#output_image > div > button > div {
    width: 100% !important;
    height: 100% !important;
    max-width: 100% !important;
    max-height: 100% !important;
    display: flex !important;
    justify_content: center !important;
    align-items: center !important;
}
#output_image img {
    object-fit: cover !important;
    width: 100% !important;
    height: 100% !important;
}
"""
dark_theme = gr.themes.Soft(primary_hue="orange").set(body_background_fill="#000000", block_background_fill="#111111", body_text_color="#EEE")

with gr.Blocks(theme=dark_theme, css=custom_css, title="Phân Loại Trái Cây") as demo:
    history_state = gr.State([])
    
    # --- OUTPUT MỚI CHO TAB THÔNG TIN MÔ HÌNH ---
    # ĐÃ LOẠI BỎ KHAI BÁO out_file_data Ở ĐÂY để tránh lỗi DuplicateBlockError
    # -------------------------------------------
    
    gr.Markdown("# Hệ thống hỗ trợ nhận diện trái cây", elem_id="header")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Nhập thông số đầu vào")
            inp_mass = gr.Slider(0, 3000, value=150, label="Khối lượng (gram)")
            inp_width = gr.Slider(0, 30.0, value=7.0, label="Chiều rộng (cm)")
            inp_height = gr.Slider(0, 40.0, value=7.0, label="Chiều cao (cm)")
            
            # ===== MENU MÀU CHI TIẾT 11 NHÓM - ĐỦ 30 LOẠI =====
            color_mapping = [
                ("1. Xanh đậm (Bơ, Chanh ta)", 0.51),
                ("2. Xanh vỏ dưa (Dưa hấu)", 0.55),
                ("3. Tím / Xanh sẫm (Dừa, Lựu, Nho, Ổi)", 0.60),
                ("4. Hỗn hợp nâu đỏ (Cà chua, Sầu riêng, Chanh dây)", 0.65),
                ("5. Vàng chanh / Xanh nhạt (Kiwi, Mận, Khế, Chanh vàng)", 0.70),
                ("6. Vàng tươi (Dứa, Bưởi, Lê, Cam)", 0.75),
                ("7. Vàng cam (Đu đủ, Táo, Mơ)", 0.79),
                ("8. Cam đỏ (Quýt, Hồng, Xoài)", 0.82),
                ("9. Hồng đỏ (Đào, Vải)", 0.85),
                ("10. Hồng tím (Thanh long, Chuối)", 0.88),
                ("11. Đỏ rực (Dâu tây, Cherry)", 0.93)
            ]
            
            inp_color = gr.Dropdown(
                choices=color_mapping, 
                value= 0.51, 
                label="Đặc điểm màu sắc",
                info="Chọn nhóm màu chứa loại quả bạn đang quan sát",
                interactive=True
            )
            
            btn_run = gr.Button("Phân tích ngay", variant="primary")

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Kết quả"):
                    with gr.Row(variant="panel"):
                        with gr.Column(scale=4):
                            out_img = gr.Image(label="Ảnh minh họa", height=320, type="filepath", interactive=False, elem_id="output_image")
                        with gr.Column(scale=6):
                            out_label_dist = gr.Label(label="Dự đoán hàng đầu", num_top_classes=5)
                    gr.Markdown("#### Phân tích chi tiết")
                    
                    out_advice = gr.Label(label="Đánh giá hệ thống", num_top_classes=0)
                    
                    # --- MỚI: Thêm ô hiển thị gợi ý ---
                    out_suggestion = gr.Markdown()
                    # ----------------------------------
                    
                    with gr.Row():
                        out_conf = gr.Label(label="Độ tin cậy (% Đồng thuận)", num_top_classes=0)
                        out_prec = gr.Label(label="Độ chính xác (Hiệu suất thực)", num_top_classes=0)
                        out_time = gr.Label(label="Thời gian xử lý", num_top_classes=0)

                with gr.TabItem("Biểu đồ"):
                    out_plot = gr.Plot()
                
                with gr.TabItem("Thống kê mô hình"):
                    gr.Markdown(f"### Độ chính xác model (Toàn cục): **{fixed_accuracy_str}**")
                    gr.Dataframe(value=counts, interactive=False)

                with gr.TabItem("Nhập file CSV"):
                    gr.Markdown("### Tải lên file CSV chứa dữ liệu")
                    with gr.Row():
                        inp_file = gr.File(label="Chọn file CSV", file_types=[".csv"])
                        with gr.Column():
                            btn_upload_run = gr.Button("Xử lý file", variant="primary")
                            out_upload_status = gr.Label(label="Trạng thái", num_top_classes=0)

                with gr.TabItem("Lịch sử"):
                    out_history_table = gr.Dataframe(headers=["STT", "M", "W", "H", "C", "Fruit", "Conf %", "Real Acc %"])
                    out_avg_conf = gr.Label(label="Thống kê nhanh (Real Acc)", num_top_classes=0)
                    with gr.Row():
                        btn_export = gr.Button("Xuất Excel", variant="secondary")
                        out_file_excel = gr.File(label="Tải về file Excel", interactive=False)

                with gr.TabItem("Thông tin mô hình"):
                    out_algo_info = gr.Markdown(generate_algo_info_markdown())
                    # --- KHAI BÁO component out_file_data Ở ĐÂY ---
                    out_file_data = gr.File(label="Tải về file dữ liệu mẫu", interactive=False)
                    # -----------------------------------------------
    
    # --- LOGIC CHO INPUT/OUTPUT TRONG TAB "THÔNG TIN MÔ HÌNH" ---
    demo.load(fn=return_data_file, inputs=None, outputs=out_file_data)
    # -------------------------------------------------------------

    inputs = [inp_mass, inp_width, inp_height, inp_color]
    # Thêm out_suggestion vào danh sách output
    outputs = [out_label_dist, out_img, out_conf, out_prec, out_advice, out_time, out_plot, out_suggestion]
    
    for inp in inputs: inp.change(fn=preview_prediction, inputs=inputs, outputs=outputs)
    
    # Cập nhật outputs cho nút run (chú ý vị trí: out_suggestion nằm trước history_table)
    btn_run.click(fn=save_prediction, inputs=inputs + [history_state], 
                  outputs=outputs + [out_history_table, out_avg_conf, history_state])
    
    # Cập nhật outputs cho btn_upload_run (thêm out_file_data)
    btn_upload_run.click(fn=process_upload, 
                         inputs=[inp_file, history_state], 
                         outputs=[history_state, out_history_table, out_upload_status, out_avg_conf, out_algo_info, out_file_data])
    
    btn_export.click(fn=export_to_excel, inputs=[history_state], outputs=[out_file_excel])

if __name__ == "__main__":
    demo.launch()