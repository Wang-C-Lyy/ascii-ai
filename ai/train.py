import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import numpy as np
import random
import os
from tqdm import tqdm

from models.vision_model import ImageEncoder
from models.language_model import TextEncoder
from models.fusion_model import MultimodalAgent
from utils.data_loader import MultimodalDataset
from config import config

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train():
    # 设置随机种子
    set_seed(config['seed'])
    
    # 初始化模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    vision_model = ImageEncoder(embedding_dim=config['vision_embedding_dim'])
    language_model = TextEncoder(embedding_dim=config['language_embedding_dim'])
    model = MultimodalAgent(
        vision_model=vision_model,
        language_model=language_model,
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    )
    
    # 移动模型到设备
    device = torch.device(config['device'])
    model = model.to(device)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    try:
        dataset = MultimodalDataset(
            image_dir=config['image_dir'],
            annotation_file=config['annotation_file'],
            tokenizer=tokenizer,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        print(f"数据集加载成功，共有{len(dataset)}个样本")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保数据集已准备好，包括图像文件和标注文件")
        return
    
    # 定义损失函数和优化器
    classification_criterion = nn.CrossEntropyLoss()
    qa_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 学习率调度器
    total_steps = len(dataloader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 创建保存模型的目录
    os.makedirs(config['model_save_path'], exist_ok=True)
    
    # 训练循环
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in progress_bar:
            # 将数据移到设备
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 - 分类任务
            classification_outputs = model(images, input_ids, attention_mask, mode="classification")
            classification_loss = classification_criterion(classification_outputs, labels)
            
            # 前向传播 - QA任务
            qa_outputs = model(images, input_ids, attention_mask, mode="qa")
            qa_outputs = qa_outputs.view(-1, qa_outputs.size(-1))
            answer_ids = answer_ids.view(-1)
            qa_loss = qa_criterion(qa_outputs, answer_ids)
            
            # 总损失
            loss = classification_loss + qa_loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), os.path.join(config['model_save_path'], f"model_epoch_{epoch+1}.pt"))
    
    print("训练完成！")

if __name__ == "__main__":
    train() 