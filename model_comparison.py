import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from io import BytesIO
import matplotlib.font_manager as fm

# 尝试查找系统上可用的中文字体
def find_chinese_font():
    """查找系统上可用的中文字体"""
    chinese_fonts = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 
                    'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei', 
                    'Heiti TC', 'Arial Unicode MS']
    
    for font in chinese_fonts:
        if font in [f.name for f in fm.fontManager.ttflist]:
            return font
    
    # 如果没有找到中文字体，使用默认字体
    print("警告: 未找到中文字体，将使用默认字体")
    return None

# 设置中文字体
chinese_font = find_chinese_font()
if chinese_font:
    plt.rcParams["font.family"] = chinese_font
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

def generate_data(n=100, noise=0.1):
    """生成更复杂的非线性分布的数据点"""
    np.random.seed(42)  # 固定随机种子，确保数据可复现
    x = np.linspace(-3, 3, n)
    y = np.sin(2*x) * np.exp(-0.5*np.abs(x)) + np.random.normal(0, noise, n)
    return x, y

# 线性回归模型
def linear_regression(x, y, iterations=100, learning_rate=0.01):
    """执行线性回归，返回每一步的参数"""
    m, b = 0, 0  # 初始参数
    history = []
    
    for i in range(iterations):
        # 计算预测值
        y_pred = m * x + b
        
        # 计算误差
        error = y_pred - y
        
        # 计算梯度
        m_gradient = np.mean(error * x)
        b_gradient = np.mean(error)
        
        # 更新参数
        m = m - learning_rate * m_gradient
        b = b - learning_rate * b_gradient
        
        # 计算损失
        loss = np.mean((y_pred - y) ** 2)
        
        # 保存当前参数和损失
        history.append((m, b, loss))
    
    return history

# 多项式回归模型
def polynomial_features(x, degree):
    """生成多项式特征"""
    X_poly = np.ones((len(x), degree + 1))
    for i in range(1, degree + 1):
        X_poly[:, i] = x ** i
    return X_poly

def polynomial_regression(x, y, degree=2, iterations=1000, learning_rate=0.01):
    """执行多项式回归，返回每一步的参数"""
    # 生成多项式特征
    X_poly = polynomial_features(x, degree)
    
    # 初始化参数
    theta = np.zeros(degree + 1)
    history = []
    
    for i in range(iterations):
        # 计算预测值
        y_pred = X_poly @ theta
        
        # 计算误差
        error = y_pred - y
        
        # 计算梯度
        gradient = (1/len(x)) * (X_poly.T @ error)
        
        # 更新参数
        theta = theta - learning_rate * gradient
        
        # 计算损失
        loss = np.mean((y_pred - y) ** 2)
        
        # 保存当前参数和损失
        history.append((theta.copy(), loss))
    
    return history

# 神经网络模型
def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU激活函数的导数"""
    return np.where(x > 0, 1, 0)

def neural_network_regression(x, y, hidden_units=20, iterations=2000, learning_rate=0.01, momentum=0.9):
    """执行优化的神经网络回归，返回每一步的参数"""
    # 数据预处理
    X = x.reshape(-1, 1)
    Y = y.reshape(-1, 1)
    
    # 固定随机种子
    np.random.seed(42)
    
    # 初始化权重
    W1 = np.random.randn(1, hidden_units) * np.sqrt(2 / 1)  # He初始化
    b1 = np.zeros((1, hidden_units))
    W2 = np.random.randn(hidden_units, hidden_units) * np.sqrt(2 / hidden_units)
    b2 = np.zeros((1, hidden_units))
    W3 = np.random.randn(hidden_units, 1) * np.sqrt(2 / hidden_units)
    b3 = np.zeros((1, 1))
    
    # 初始化动量项
    vW1 = np.zeros_like(W1)
    vb1 = np.zeros_like(b1)
    vW2 = np.zeros_like(W2)
    vb2 = np.zeros_like(b2)
    vW3 = np.zeros_like(W3)
    vb3 = np.zeros_like(b3)
    
    history = []
    
    for i in range(iterations):
        # 前向传播
        Z1 = np.dot(X, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        A3 = Z3  # 输出层不使用激活函数（回归问题）
        
        # 计算误差
        error = A3 - Y
        
        # 反向传播
        dZ3 = error
        dW3 = np.dot(A2.T, dZ3) / len(X)
        db3 = np.sum(dZ3, axis=0, keepdims=True) / len(X)
        
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2) / len(X)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / len(X)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X.T, dZ1) / len(X)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / len(X)
        
        # 使用动量更新权重
        vW1 = momentum * vW1 + learning_rate * dW1
        vb1 = momentum * vb1 + learning_rate * db1
        vW2 = momentum * vW2 + learning_rate * dW2
        vb2 = momentum * vb2 + learning_rate * db2
        vW3 = momentum * vW3 + learning_rate * dW3
        vb3 = momentum * vb3 + learning_rate * db3
        
        W1 -= vW1
        b1 -= vb1
        W2 -= vW2
        b2 -= vb2
        W3 -= vW3
        b3 -= vb3
        
        # 计算损失
        loss = np.mean(np.square(error))
        
        # 每20次迭代保存一次状态，减少GIF大小
        if i % 20 == 0 or i == iterations - 1:
            history.append((W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy(), loss))
    
    return history

# 预测函数
def linear_predict(x, m, b):
    """线性回归预测"""
    return m * x + b

def polynomial_predict(x, theta, degree):
    """多项式回归预测"""
    X_poly = polynomial_features(x, degree)
    return X_poly @ theta

def nn_predict(x, W1, b1, W2, b2, W3, b3):
    """神经网络预测"""
    X = x.reshape(-1, 1)
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    return Z3.flatten()

# 创建动画函数
def create_animation(x, y, history, predict_func, model_name, degree=None, output_path=None):
    """创建并保存动画"""
    if output_path is None:
        output_path = f'{model_name}_regression_fit.gif'
    
    frames = []
    x_plot = np.linspace(min(x)-1, max(x)+1, 200)
    
    for i, params in enumerate(history):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制数据点
        ax.scatter(x, y, color='blue', label='数据点')
        
        # 绘制真实函数曲线
        true_y = np.sin(2*x_plot) * np.exp(-0.5*np.abs(x_plot))
        ax.plot(x_plot, true_y, color='green', linestyle='--', label='真实函数')
        
        # 绘制当前拟合曲线
        if model_name == '线性':
            y_plot = predict_func(x_plot, *params[:2])
            ax.plot(x_plot, y_plot, color='red', label=f'线性拟合')
            ax.set_title(f'线性回归拟合过程 (迭代: {i+1}/{len(history)}, MSE: {params[2]:.6f})')
        elif model_name == '多项式':
            y_plot = predict_func(x_plot, params[0], degree)
            ax.plot(x_plot, y_plot, color='red', label=f'{degree}次多项式拟合')
            ax.set_title(f'{degree}次多项式回归拟合 (迭代: {i+1}/{len(history)}, MSE: {params[1]:.6f})')
        elif model_name == '神经网络':
            y_plot = predict_func(x_plot, *params[:-1])
            ax.plot(x_plot, y_plot, color='red', label='神经网络拟合')
            ax.set_title(f'神经网络回归拟合 (迭代: {i*20+1}/{len(history)*20}, MSE: {params[-1]:.6f})')
        
        # 设置标题和标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(min(y)-0.5, max(y)+0.5)
        
        # 保存当前帧
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = imageio.imread(buf)
        frames.append(img)
        plt.close()
    
    # 保存为GIF
    imageio.mimsave(output_path, frames, fps=10)
    print(f"{model_name}回归GIF已保存至: {output_path}")

def main():
    # 生成数据
    x, y = generate_data()
    
    # 1. 线性回归
    print("训练线性回归模型...")
    linear_history = linear_regression(x, y, iterations=500, learning_rate=0.01)
    create_animation(x, y, linear_history, linear_predict, "线性")
    
    # 2. 多项式回归 (3次)
    print("训练3次多项式回归模型...")
    poly3_history = polynomial_regression(x, y, degree=3, iterations=1000, learning_rate=0.01)
    create_animation(x, y, poly3_history, polynomial_predict, "多项式", degree=3)
    
    # 3. 多项式回归 (5次)
    print("训练5次多项式回归模型...")
    poly5_history = polynomial_regression(x, y, degree=5, iterations=1000, learning_rate=0.005)
    create_animation(x, y, poly5_history, polynomial_predict, "多项式", degree=5)
    
    # 4. 神经网络
    print("训练神经网络模型...")
    nn_history = neural_network_regression(x, y, hidden_units=20, iterations=2000, learning_rate=0.01)
    create_animation(x, y, nn_history, nn_predict, "神经网络")
    
    print("\n所有模型训练和动画生成完成！")

if __name__ == "__main__":
    main()    