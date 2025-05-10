import matplotlib.pyplot as plt
import numpy as np

# 定义学习率、训练时间和最佳验证精度数据
learning_rates_sgd = [0.001, 0.002, 0.0008, 0.0009, 0.0012]
training_times_sgd = [11 * 60 + 3, 10 * 60 + 39, 10 * 60 + 46, 10 * 60 + 53, 10 * 60 + 60]  # 转为秒
best_val_acc_sgd = [0.956140, 0.950877, 0.955614, 0.966667, 0.954386]

learning_rates_adam = [0.001, 0.002, 0.0009, 0.0012, 0.0008]
training_times_adam = [11 * 60 + 12, 11 * 60 + 6, 10 * 60 + 56, 10 * 60 + 41, 10 * 60 + 51]  # 转为秒
best_val_acc_adam = [0.945614, 0.929825, 0.954386, 0.949123, 0.945614]

# 设置柱状图的宽度和位置
bar_width = 0.35
x = np.arange(len(learning_rates_sgd))

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 绘制训练时间的柱状图
ax1.bar(x - bar_width / 2, training_times_sgd, width=bar_width, label='SGD', color='blue')
ax1.bar(x + bar_width / 2, training_times_adam, width=bar_width, label='Adam', color='orange')
ax1.set_title('Training Time Comparison')
ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Training Time (seconds)')
ax1.set_xticks(x)  # 设置 x 轴位置
ax1.set_xticklabels(learning_rates_sgd)  # 设置 x 轴标签为学习率
ax1.legend()
ax1.grid()

# 绘制最佳验证精度的柱状图
ax2.bar(x - bar_width / 2, best_val_acc_sgd, width=bar_width, label='SGD', color='blue')
ax2.bar(x + bar_width / 2, best_val_acc_adam, width=bar_width, label='Adam', color='orange')
ax2.set_title('Best Validation Accuracy Comparison')
ax2.set_xlabel('Learning Rate')
ax2.set_ylabel('Best Validation Accuracy')
ax2.set_xticks(x)  # 设置 x 轴位置
ax2.set_xticklabels(learning_rates_sgd)  # 设置 x 轴标签
ax2.legend()
ax2.grid()

# 调整布局
plt.tight_layout()

# 保存图像到指定路径
save_path = 'F:\\桌面\\大三下\\神经网络\\EX2\data_picture\\best_val_acc_sgd_compare1.png'
plt.savefig(save_path, dpi=300)  # 保存为 PNG 格式，设置分辨率为 300 DPI
print(f"图像已保存到: {save_path}")

# 显示图形
plt.show()