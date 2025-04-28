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
# 可视化数字 0-9，每行显示十个相同数字的不同手写图像
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
    # 找到所有标签为 i 的图像索引
    indices = np.where(y == i)[0]
    for j in range(10):
        # 随机选择一个标签为 i 的图像索引
        index = np.random.choice(indices)
        image = X[index].reshape(28, 28)  # 将一维数据转换为 28x28 的图像
        # 显示图像
        ax = axes[i, j]
        ax.imshow(image, cmap='gray')
        ax.axis('off')  # 不显示坐标轴
        # ax.set_title(f'数字 {i}')
plt.tight_layout()
# 保存图片为mnist.png
plt.savefig('mnist.png', dpi=300, bbox_inches='tight')
plt.show()