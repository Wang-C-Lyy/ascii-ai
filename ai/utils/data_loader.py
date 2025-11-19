import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, annotation_file, tokenizer, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        
        # 加载标注数据
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
            
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.image_dir, item['image_filename'])
        question = item['question']
        answer = item['answer']
        label = item.get('label', 0)  # 如果有分类标签
        
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 处理文本
        text_encoding = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        # 处理答案文本（用于训练）
        answer_encoding = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        return {
            'image': image,
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'answer_ids': answer_encoding['input_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        } 