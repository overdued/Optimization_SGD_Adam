import matplotlib.pyplot as plt
import os

# 定义文件路径
data1_path = "F:\\桌面\\大三下\\神经网络\\EX2\\data_SDG\\training_log5.txt"
data2_path = "F:\\桌面\\大三下\\神经网络\\EX2\\data_adam\\training_log5.txt"

# 读取数据函数
def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    data = []
    for line in lines[1:]:  # 跳过表头
        if line.strip() and not any(keyword in line for keyword in ["学习率", "训练完成时间", "最佳验证精度"]):
            data.append(line.strip().split("\t"))
    
    epochs = [int(row[0]) for row in data]
    train_loss = [float(row[1]) for row in data]
    val_loss = [float(row[2]) for row in data]
    train_acc = [float(row[3]) for row in data]
    val_acc = [float(row[4]) for row in data]
    
    return epochs, train_loss, val_loss, train_acc, val_acc
    
    return epochs, train_loss, val_loss, train_acc, val_acc
# 读取两个数据集
epochs1, train_loss1, val_loss1, train_acc1, val_acc1 = read_data(data1_path)
epochs2, train_loss2, val_loss2, train_acc2, val_acc2 = read_data(data2_path)

# 创建保存目录
save_dir = "F:\\桌面\\大三下\\神经网络\\EX2\\data_picture"
os.makedirs(save_dir, exist_ok=True)

# 绘制损失曲线
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs1, train_loss1, label='Train Loss Group 1', color='blue')
plt.plot(epochs2, train_loss2, label='Train Loss Group 2', color='green')
plt.title('Train Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, max(max(epochs1), max(epochs2)) + 1))
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(epochs1, val_loss1, label='Validation Loss Group 1', color='orange')
plt.plot(epochs2, val_loss2, label='Validation Loss Group 2', color='red')
plt.title('Validation Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(range(0, max(max(epochs1), max(epochs2)) + 1))
plt.legend()
plt.grid()

# 绘制准确率曲线
plt.subplot(2, 2, 3)
plt.plot(epochs1, train_acc1, label='Train Accuracy Group 1', color='blue')
plt.plot(epochs2, train_acc2, label='Train Accuracy Group 2', color='green')
plt.title('Train Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(0, max(max(epochs1), max(epochs2)) + 1))
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(epochs1, val_acc1, label='Validation Accuracy Group 1', color='orange')
plt.plot(epochs2, val_acc2, label='Validation Accuracy Group 2', color='red')
plt.title('Validation Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(range(0, max(max(epochs1), max(epochs2)) + 1))
plt.legend()
plt.grid()

plt.tight_layout()

# 保存图表到指定路径
plt.savefig(os.path.join(save_dir, 'training_comparison5.png'))
print(f"图表已保存到: {os.path.join(save_dir, 'training_comparison2.png')}")
plt.show()