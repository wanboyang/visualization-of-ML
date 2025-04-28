import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
from sklearn.datasets import load_iris

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载iris数据集
iris_data = load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = iris_data.target
species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris['species'] = iris['species'].map(species_mapping)

# 执行PCA降维
pca = decomposition.PCA(n_components=2)
X = pca.fit_transform(iris.iloc[:, :-1].values)

# 创建结果DataFrame
iris_pca = pd.DataFrame(data=X, columns=['主成分1', '主成分2'])
iris_pca['species'] = iris['species']

# 打印PCA结果
print("\n===== PCA分析结果 =====")

# 1. 解释方差比例
explained_variance = pca.explained_variance_ratio_
print("\n1. 主成分解释方差比例:")
for i, ratio in enumerate(explained_variance, 1):
    print(f"   主成分{i}: {ratio:.4f} ({ratio*100:.2f}%)")
print(f"   累计解释方差: {sum(explained_variance):.4f} ({sum(explained_variance)*100:.2f}%)")

# 2. 特征载荷矩阵（主成分系数）
print("\n2. 特征载荷矩阵（特征对主成分的贡献）:")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=['主成分1', '主成分2'],
    index=iris.columns[:-1]  # 排除species列
)
print(loadings.round(4))

# 3. 展示前10个样本的PCA转换结果
print("\n3. PCA转换后的样本数据（前10个样本）:")
print(iris_pca.head(10).round(4))

# 使用seaborn绘制散点图
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=iris_pca,
    x='主成分1',
    y='主成分2',
    hue='species',
    palette=['blue', 'green', 'red'],
    s=80,
    alpha=0.8,
    edgecolor='black'
)

# 添加标题和标签
plt.title('鸢尾花数据集PCA降维结果', fontsize=15)
plt.xlabel(f'主成分1 (解释方差: {explained_variance[0]:.2%})', fontsize=12)
plt.ylabel(f'主成分2 (解释方差: {explained_variance[1]:.2%})', fontsize=12)

# 添加特征载荷箭头
components = pca.components_
feature_names = iris.columns[:-1]
scale = max(iris_pca['主成分1'].abs().max(), iris_pca['主成分2'].abs().max()) * 1.2

for i, feature in enumerate(feature_names):
    plt.arrow(
        0, 0,
        components[0, i] * scale,
        components[1, i] * scale,
        head_width=0.1,
        head_length=0.1,
        fc='black',
        ec='black',
        linestyle='--',
        alpha=0.7
    )
    plt.text(
        components[0, i] * scale * 1.1,
        components[1, i] * scale * 1.1,
        feature,
        fontsize=10,
        fontweight='bold'
    )

# 保存图片
plt.savefig('iris_pca_seaborn.png', dpi=300, bbox_inches='tight')
print("\n图片已保存为 'iris_pca_seaborn.png'")

# 显示图形
plt.show()