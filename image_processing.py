import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# 1. 讀取圖像
image = cv2.imread(r'C:\Users\IVAN\Downloads\image.jpg')  # 讀取輸入圖像
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將圖像從 BGR 轉換為 RGB 格式

# 2. 轉換為 HSV 空間 (HSV 用於顏色範圍檢測)
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# 3. 平滑處理 (可選，用於減少 HSV 檢測中的噪聲)
hsv_blur = cv2.GaussianBlur(hsv_image, (5, 5), 0)

# 4. 使用 HSV 顏色閾值識別馬路區域
# 定義 HSV 範圍
lower_hsv = np.array([0, 0, 30])  # HSV 下限
upper_hsv = np.array([180, 100, 100])  # HSV 上限
mask = cv2.inRange(hsv_blur, lower_hsv, upper_hsv)  # 創建二值遮罩，識別 HSV 範圍內的像素

# 5. HSV 初步塗色處理
colored_image = image.copy()
colored_image[mask == 255] = [0, 255, 0]  # 使用綠色標記初步識別的馬路區域

# 6. LBP 特徵提取 (對灰階圖像進行處理，用於紋理分析)
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 將圖像轉換為灰階
radius = 1  # LBP 半徑
n_points = 8 * radius  # LBP 的取樣點數量
lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')  # 計算 LBP 特徵

# 7. 定義 LBP 模板和計算 L^1 範數距離
# 設定馬路目標的 LBP 模板值
road_template = 3  # 假設目標區域的 LBP 值為 3
lbp_distance = np.abs(lbp - road_template)  # 計算每個像素的 LBP 值與模板的 L^1 範數距離

# 8. 根據 L^1 範數距離細化馬路標記 (結合 HSV + LBP)
threshold = 2  # 範數距離的閾值，用於過濾非目標區域
final_mask = (mask == 255) & (lbp_distance <= threshold)  # 結合 HSV 遮罩和 LBP 範數距離

# 更新塗色圖像 (LBP + HSV 綜合結果)
final_colored_image = image.copy()
final_colored_image[final_mask] = [0, 0, 255]  # 使用紅色標記改進後的馬路區域

# 9. 灰階直方圖 (用於圖像灰度特徵分析)
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # 計算灰階直方圖

# 10. 顯示結果
plt.figure(figsize=(15, 10))

# 原始圖像
plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# LBP 特徵圖
plt.subplot(2, 3, 2)
plt.imshow(lbp, cmap='gray')
plt.title('LBP Features')
plt.axis('off')

# 灰階直方圖
plt.subplot(2, 3, 3)
plt.plot(hist, color='black')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 初步馬路區域標記圖 (基於 HSV 閾值檢測)
plt.subplot(2, 3, 4)
plt.imshow(colored_image)
plt.title('(HSV & BFS)')
plt.axis('off')

# 基於 LBP + L^1 範數的改進標記圖
plt.subplot(2, 3, 5)
plt.imshow(final_colored_image)
plt.title('(HSV&BFS + LBP&1 norm distance)')
plt.axis('off')



# 顯示圖像
plt.tight_layout()
plt.show()
