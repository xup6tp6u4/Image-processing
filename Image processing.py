import cv2
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt

# 1. 讀取影像並轉換為灰階圖像
img = cv2.imread('D:/input_image.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 轉換為灰階圖像

# 2. Sobel邊緣檢測
def sobel_edge_detection(image):
    # Sobel kernel
    sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 卷積
    sobel_x = cv2.filter2D(image, -1, sobel_x_kernel)
    sobel_y = cv2.filter2D(image, -1, sobel_y_kernel)

    # 計算邊緣強度
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
    
    return sobel_magnitude

sobel_img = sobel_edge_detection(gray_img)

# 3. LBP 編碼
def manual_lbp(image, radius=1, n_points=8):
    lbp_img = np.zeros_like(image, dtype=np.uint8)
    for i in range(radius, image.shape[0] - radius):
        for j in range(radius, image.shape[1] - radius):
            center = image[i, j]
            binary_pattern = 0
            for k in range(n_points):
                # 計算鄰居點的位置
                theta = 2 * np.pi * k / n_points
                y = int(i + radius * np.sin(theta))
                x = int(j + radius * np.cos(theta))
                # 比較中心像素與鄰居像素
                binary_pattern <<= 1
                if image[y, x] >= center:
                    binary_pattern |= 1
            lbp_img[i, j] = binary_pattern
    return lbp_img

lbp_img = manual_lbp(sobel_img)

# 計算直方圖
def manual_histogram(image, bins, range_):
    hist = np.zeros(bins, dtype=np.float32)
    bin_width = (range_[1] - range_[0]) / bins
    for value in image.ravel():
        bin_index = int((value - range_[0]) / bin_width)
        if 0 <= bin_index < bins:
            hist[bin_index] += 1
    return hist

n_points = 8  # LBP的點數
histogram = manual_histogram(lbp_img, bins=n_points + 2, range_=(0, n_points + 2))
histogram /= histogram.sum()  # 正規化直方圖

# 4. 搜尋：根據直方圖找相似數值
search result = np.zeros_like(gray_img, dtype=np.uint8)
threshold = 0.1  # 設定搜尋的閾值

# 計算每個區域的直方圖並進行比對
for i in range(0, gray_img.shape[0] - 3, 3):  # 以3x3區塊進行搜尋
    for j in range(0, gray_img.shape[1] - 3, 3):
        # 擷取區域
        patch = lbp_img[i:i+3, j:j+3]
        # 計算區域的LBP直方圖
        patch_histogram = manual_histogram(patch, bins=n_points + 2, range_=(0, n_points + 2))
        patch_histogram /= patch_histogram.sum()  # 正規化直方圖
        
        # 計算每個區域的直方圖與相鄰區塊直方圖的歐氏距離
        for di in range(0, gray_img.shape[0] - 3, 3):
            for dj in range(0, gray_img.shape[1] - 3, 3):
                # 擷取鄰近區域
                neighbor_patch = lbp_img[di:di+3, dj:dj+3]
                neighbor_histogram = manual_histogram(neighbor_patch, bins=n_points + 2, range_=(0, n_points + 2))
                neighbor_histogram /= neighbor_histogram.sum()  # 正規化直方圖
                
                # 計算直方圖的歐氏距離
                distance = np.linalg.norm(patch_histogram - neighbor_histogram)
                
                # 如果距離小於閾值，則標註為相似
                if distance < threshold:
                    search_result[i:i+3, j:j+3] = 255  # 標註為白色區域

# 5. 塗色標註 (使用標籤)
labeled_img = measure.label(search_result)  # 使用連通元件標註
colored_img = cv2.applyColorMap(labeled_img.astype(np.uint8) * 50, cv2.COLORMAP_JET)  # 塗上顏色

# 顯示結果
plt.figure(figsize=(10, 8))

# 顯示原始圖片
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# 顯示Sobel邊緣檢測結果
plt.subplot(1, 3, 2)
plt.imshow(sobel_img, cmap='gray')
plt.title('Sobel Edge Detection')

# 顯示塗色標註結果
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
plt.title('Search & Labeling')

plt.tight_layout()
plt.show()
