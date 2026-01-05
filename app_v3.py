import streamlit as st
import cv2
import numpy as np
import pandas as pd
from io import BytesIO
import zipfile

# --- 页面配置 ---
st.set_page_config(page_title="细胞划痕分析 Pro (可视化修复版)", layout="wide")

st.title("🔬 细胞划痕分析 Pro (T0对比)")

# --- 核心算法 ---
def analyze_scratch(image_file, sigma=15, thresh_offset=0, min_area=1000, 
                    keep_only_largest=True, line_thickness=2):
    
    # 1. 读取
    # 每次读取前重置指针，防止多次调用报错
    image_file.seek(0)
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    # 安全检查：防止空文件报错
    if original_img is None:
        return None, None, None, 0, 0, 0

    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # 2. 预处理
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.uint8(np.absolute(cv2.magnitude(sobel_x, sobel_y)))
    
    k_size = (sigma * 2) + 1
    blurred_mag = cv2.GaussianBlur(magnitude, (k_size, k_size), 0)
    
    otsu_thresh, _ = cv2.threshold(blurred_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final_thresh = max(0, min(255, otsu_thresh + thresh_offset))
    _, mask = cv2.threshold(blurred_mag, final_thresh, 255, cv2.THRESH_BINARY)
    gap_mask = cv2.bitwise_not(mask)

    # 3. 轮廓筛选
    contours, _ = cv2.findContours(gap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if keep_only_largest and len(contours) > 0:
            valid_contours = [contours[0]]
        else:
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # 4. 绘图与计算
    clean_mask = np.zeros_like(gap_mask)
    cv2.drawContours(clean_mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    annotated_img = original_img.copy()
    cv2.drawContours(annotated_img, valid_contours, -1, (0, 255, 255), thickness=line_thickness, lineType=cv2.LINE_AA)

    height, width = clean_mask.shape
    total_pixels = height * width
    gap_pixels = cv2.countNonZero(clean_mask)
    
    area_ratio = (gap_pixels / total_pixels) * 100
    avg_width_px = gap_pixels / height

    return original_img, clean_mask, annotated_img, area_ratio, avg_width_px, gap_pixels

# --- 侧边栏 ---
with st.sidebar:
    st.header("1. 图片上传")
    uploaded_files = st.file_uploader("请上传同一组实验的所有图片", type=['jpg', 'png', 'tif'], accept_multiple_files=True)

    # 基准选择
    baseline_file = None
    if uploaded_files:
        file_names = [f.name for f in uploaded_files]
        st.header("2. 设定基准 (T0)")
        selected_baseline_name = st.selectbox(
            "请选择 0 小时 (T0) 图片:", 
            options=file_names
        )
        baseline_file = next(f for f in uploaded_files if f.name == selected_baseline_name)

    st.header("3. 算法微调 (实时预览)")
    smart_mode = st.checkbox("✅ 只保留主划痕 (推荐)", value=True)
    p_sigma = st.slider("纹理模糊度", 1, 50, 15)
    p_thresh = st.slider("阈值修正", -50, 50, 0)
    p_min_area = st.number_input("最小面积过滤", value=1000)
    line_thick = st.slider("描边粗细", 1, 5, 2)

# --- 主逻辑 ---
if uploaded_files and baseline_file:
    
    # === 1. 实时预览区域 (修复回来的部分！) ===
    st.subheader(f"👁️ 参数调试预览 (当前显示: {baseline_file.name})")
    
    # 分析选中的 T0 图片
    # 注意：这里调用函数用于显示，下面批量分析时会再次调用
    _, t0_mask, t0_anno, t0_area, t0_width, t0_pixels = analyze_scratch(
        baseline_file, p_sigma, p_thresh, p_min_area, smart_mode, line_thick
    )
    
    # 显示三栏布局：原图描边 | 掩膜 | 数据
    col_p1, col_p2, col_p3 = st.columns([2, 2, 1])
    
    with col_p1:
        st.image(t0_anno, channels="BGR", caption="识别结果 (黄色描边)", use_container_width=True)
    with col_p2:
        st.image(t0_mask, caption="计算掩膜 (Mask)", use_container_width=True)
    with col_p3:
        st.info("调整左侧滑块，\n直到此处识别准确。")
        st.metric("T0 面积占比", f"{t0_area:.2f}%")
        st.metric("T0 初始宽度", f"{t0_width:.1f} px")

    st.divider()

    # === 2. 批量处理区域 ===
    st.subheader("🚀 批量分析")
    if st.button(f"参数满意，开始基于 {baseline_file.name} 分析所有图片"):
        results = []
        zip_buffer = BytesIO()
        my_bar = st.progress(0)
        
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for i, file in enumerate(uploaded_files):
                
                # 运行分析
                _, _, res_img, res_area, res_width, res_pixels = analyze_scratch(
                    file, p_sigma, p_thresh, p_min_area, smart_mode, line_thick
                )
                
                if res_img is None: continue # 跳过坏图

                # 计算愈合率
                if t0_pixels > 0:
                    healing_rate = ((t0_pixels - res_pixels) / t0_pixels) * 100
                else:
                    healing_rate = 0.0
                
                # 存数据
                results.append({
                    "文件名": file.name,
                    "划痕面积占比(%)": round(res_area, 2),
                    "平均宽度(px)": round(res_width, 1),
                    "愈合率(%)": round(healing_rate, 2),
                    "相对迁移距离(px)": round(t0_width - res_width, 1)
                })
                
                # 存图片
                _, img_encoded = cv2.imencode('.jpg', res_img)
                zf.writestr(f"Proc_{file.name}", img_encoded.tobytes())
                
                my_bar.progress((i + 1) / len(uploaded_files))
        
        # 结果展示
        df = pd.DataFrame(results).sort_values(by="文件名")
        
        st.success("✅ 分析完成！")
        
        # 高亮表格
        st.dataframe(
            df.style.highlight_max(axis=0, subset=["愈合率(%)"], color="#90EE90"), 
            use_container_width=True
        )
        
        # 简单图表
        st.line_chart(df, x="文件名", y="愈合率(%)")
        
        # 下载
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button("📄 下载数据表 (CSV)", df.to_csv(index=False).encode('utf-8-sig'), "report.csv", "text/csv")
        with col_d2:
            st.download_button("🖼️ 下载图片包 (ZIP)", zip_buffer.getvalue(), "images.zip", "application/zip")

elif not uploaded_files:
    st.info("👈 请先在左侧上传图片")