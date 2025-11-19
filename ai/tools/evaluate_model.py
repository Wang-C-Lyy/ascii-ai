import torch
import json
import argparse
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

import sys
sys.path.append('..')  # 添加父目录到路径

from models.vision_model import ImageEncoder
from models.language_model import TextEncoder
from models.fusion_model import MultimodalAgent
from utils.data_loader import MultimodalDataset
from transformers import BertTokenizer
from torchvision import transforms
from config import config

def evaluate_model(model_path, test_data_path, batch_size=32):
    # 加载模型
    device = torch.device(config['device'])
    
    vision_model = ImageEncoder(embedding_dim=config['vision_embedding_dim'])
    language_model = TextEncoder(embedding_dim=config['language_embedding_dim'])
    model = MultimodalAgent(
        vision_model=vision_model,
        language_model=language_model,
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 数据预处理
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载测试数据
    test_dataset = MultimodalDataset(
        image_dir=os.path.dirname(test_data_path) + '/images',
        annotation_file=test_data_path,
        tokenizer=tokenizer,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 评估
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估中"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # 分类预测
            outputs = model(images, input_ids, attention_mask, mode="classification")
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    parser = argparse.ArgumentParser(description="评估多模态智能体模型")
    parser.add_argument("--model_path", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--test_data", type=str, required=True, help="测试数据路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误：模型文件 {args.model_path} 不存在")
        return
    
    if not os.path.exists(args.test_data):
        print(f"错误：测试数据文件 {args.test_data} 不存在")
        return
    
    evaluate_model(args.model_path, args.test_data, args.batch_size)

if __name__ == "__main__":
    main() 