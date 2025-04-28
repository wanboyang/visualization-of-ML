import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import matplotlib.font_manager as fm

# 尝试查找系统上可用的中文字体
def find_chinese_font():
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC',
                    'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei',
                    'Heiti TC', 'Arial Unicode MS']
    for font in chinese_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    print("警告: 未找到中文字体，将使用默认字体")
    return None

# 设置中文字体
chinese_font = find_chinese_font()
if chinese_font:
    plt.rcParams["font.family"] = chinese_font
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载 MNIST 数据集
mnist = fetch_openml('mnist_784', version=1, cache=True)
X, y = mnist.data, mnist.target

# 将标签转换为整数类型
y = y.astype(int)

# 将 X 转换为 numpy.ndarray 类型
X = np.array(X)

# 选择要可视化的图像索引
image_index = 0
image_data = X[image_index].reshape(28, 28)
true_label = y[image_index]

# 统计像素值分布
pixel_values = image_data.flatten()
histogram, bin_edges = np.histogram(pixel_values, bins=256, range=[0, 256])

# 创建一个更大的图像来同时显示原图、像素值热图和像素分布
fig = plt.figure(figsize=(18, 5))

# 显示原图
ax1 = fig.add_subplot(131)
ax1.imshow(image_data, cmap='gray')
ax1.set_title(f'MNIST 数字: {true_label}')
ax1.axis('off')

# 显示像素值热图
ax2 = fig.add_subplot(132)
im = ax2.imshow(image_data, cmap='viridis')
ax2.set_title('像素值热图')
ax2.axis('off')

# 在热图上显示每个像素的值
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        # 只显示非零像素值，避免画面过于拥挤
        if image_data[i, j] > 0:
            text_color = 'white' if image_data[i, j] > 128 else 'black'
            ax2.text(j, i, f'{image_data[i, j]:.0f}',
                    ha='center', va='center',
                    color=text_color, fontsize=5)

# 添加颜色条显示像素值范围
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('像素强度')

# 显示像素值分布
ax3 = fig.add_subplot(133)
ax3.bar(bin_edges[:-1], histogram, width=1.0, color='skyblue')
ax3.set_title('像素值分布')
ax3.set_xlabel('像素值')
ax3.set_ylabel('像素数量')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# 统计信息
total_pixels = image_data.size
non_zero_pixels = np.count_nonzero(image_data)
zero_percentage = (1 - non_zero_pixels / total_pixels) * 100

# 在分布图上添加统计信息
stats_text = f'总像素: {total_pixels}\n'
stats_text += f'非零像素: {non_zero_pixels}\n'
stats_text += f'空白区域比例: {zero_percentage:.2f}%'
ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('mnist_pixel_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印详细统计信息
print(f"MNIST图像 {image_index} 的像素统计信息:")
print(f" - 像素值范围: {np.min(image_data)} 到 {np.max(image_data)}")
print(f" - 平均像素值: {np.mean(image_data):.2f}")
print(f" - 像素值中位数: {np.median(image_data):.2f}")
print(f" - 像素值标准差: {np.std(image_data):.2f}")