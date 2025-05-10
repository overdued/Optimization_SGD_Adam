import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.optim import lr_scheduler  
from torchvision import datasets, transforms, models  
from torch.utils.data import random_split  
import os  
import time  
import copy  
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图

# 设置数据目录  
data_dir = "F:\\桌面\\大三下\\神经网络\\flower_dataset\\flower_dataset"  

# 数据增强和归一化  
data_transforms = transforms.Compose([  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(20),      
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])  

# 加载完整数据集  
full_dataset = datasets.ImageFolder(data_dir, data_transforms)  

# 自动拆分为80%训练集和20%验证集  
train_size = int(0.8 * len(full_dataset))  
val_size = len(full_dataset) - train_size  

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])  

# 使用 DataLoader 处理训练和验证集  
batch_size = 32  
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)  

dataloaders = {'train': train_loader, 'val': val_loader}  
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}  

# 获取数据集类别名称  
class_names = full_dataset.classes  

# 加载预训练模型并修改最后一层  
model = models.resnet18(pretrained=True)  

# 修改模型的最后一个全连接层  
num_classes = 5  
model.fc = nn.Linear(model.fc.in_features, num_classes)  

# 定义损失函数  
criterion = nn.CrossEntropyLoss()  

# 定义优化器  
#optimizer = optim.Adam(model.parameters(), lr=0.001)  
optimizer = optim.SGD(model.parameters(), lr=0.0012, momentum=0.9)  # 学习率设置为0.01，动量为0.9
# 学习率调度器
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0)  
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 学习率设置为0.01，动量为0.9
# 训练模型的函数  
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):  
    since = time.time()  
    best_model_wts = copy.deepcopy(model.state_dict())  
    best_acc = 0.0  

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model = model.to(device)  

    # 创建保存损失和精度的文件
    log_file_path = "F:\\桌面\\大三下\\神经网络\\EX2\\data_SDG\\training_log5.txt"
    with open(log_file_path, "w") as log_file:
        log_file.write("Epoch\tTrain Loss\tVal Loss\tTrain Acc\tVal Acc\n")

        # 用于存储每个epoch的损失和精度
        train_loss_history = [1]
        val_loss_history = [1]
        train_acc_history = [1]
        val_acc_history = [1]

        for epoch in range(num_epochs):  
            print(f'轮次 {epoch}/{num_epochs - 1}')  
            print('-' * 10)  
            current_lr = optimizer.param_groups[0]['lr']  
            print(f'学习率: {current_lr:.6f}')  

            for phase in ['train', 'val']:  
                if phase == 'train':  
                    model.train()  
                else:  
                    model.eval()  

                running_loss = 0.0  
                running_corrects = 0  

                for inputs, labels in dataloaders[phase]:  
                    inputs = inputs.to(device)  
                    labels = labels.to(device)  

                    optimizer.zero_grad()  

                    with torch.set_grad_enabled(phase == 'train'):  
                        outputs = model(inputs)  
                        _, preds = torch.max(outputs, 1)  
                        loss = criterion(outputs, labels)  

                        if phase == 'train':  
                            loss.backward()  
                            optimizer.step()   

                    running_loss += loss.item() * inputs.size(0)  
                    running_corrects += torch.sum(preds == labels.data)  

                if phase == 'train':  
                    scheduler.step()  

                epoch_loss = running_loss / dataset_sizes[phase]  
                epoch_acc = running_corrects.double() / dataset_sizes[phase]  

                print(f'{phase} 损失: {epoch_loss:.4f} 精度: {epoch_acc:.4f}')  

                # 记录损失和精度
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc.item())
                else:
                    val_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc.item())

                # 记录训练和验证的损失和精度到文件
                log_file.write(f"{epoch}\t{train_loss_history[-1]:.4f}\t{val_loss_history[-1]:.4f}\t{train_acc_history[-1]:.4f}\t{val_acc_history[-1]:.4f}\n")

                # 保存最佳模型  
                if phase == 'val' and epoch_acc > best_acc:  
                    best_acc = epoch_acc  
                    best_model_wts = copy.deepcopy(model.state_dict())  
                    save_dir = "F:\\桌面\\大三下\\神经网络\EX2\\best_model_SDG"  
                    os.makedirs(save_dir, exist_ok=True)  
                    model_save_path = os.path.join(save_dir, '0.0012.pth')  
                    torch.save(model.state_dict(), model_save_path)  

        print()  

    time_elapsed = time.time() - since  
    print(f'训练完成时间: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')  
    print(f'最佳验证精度: {best_acc:4f}')  
    return model  

# 添加主程序入口 
if __name__ == '__main__':  
    # 训练模型  
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    model = model.to(device)  
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)  