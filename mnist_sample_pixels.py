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

# 创建一个大的图像来同时显示原图和像素值
fig = plt.figure(figsize=(12, 6))

# 显示原图
ax1 = fig.add_subplot(121)
ax1.imshow(image_data, cmap='gray')
ax1.set_title(f'MNIST 数字: {true_label}')
ax1.axis('off')

# 显示像素值热图
ax2 = fig.add_subplot(122)
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

plt.tight_layout()
plt.savefig('mnist_pixel_values.png', dpi=300, bbox_inches='tight')
plt.show()