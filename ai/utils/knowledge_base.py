import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

class KnowledgeBase:
    def __init__(self, knowledge_file, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 加载知识库
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            self.knowledge = json.load(f)
        
        # 创建索引
        self.build_index()
        
    def build_index(self):
        """构建向量索引"""
        self.model.eval()
        embeddings = []
        
        for item in self.knowledge:
            # 获取问题的嵌入表示
            inputs = self.tokenizer(item['question'], return_tensors='pt', padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(embedding)
        
        # 将嵌入转换为numpy数组
        self.embeddings = np.vstack(embeddings)
        
        # 创建简单的向量索引（如果有faiss，可以替换为faiss索引）
        self.index = self.embeddings
        
    def search(self, query, k=5):
        """搜索最相关的知识"""
        # 获取查询的嵌入表示
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 计算余弦相似度
        similarities = np.dot(self.index, query_embedding.T) / (
            np.linalg.norm(self.index, axis=1, keepdims=True) * 
            np.linalg.norm(query_embedding, keepdims=True)
        )
        
        # 获取最相似的k个结果
        indices = np.argsort(-similarities.flatten())[:k]
        
        # 返回最相关的知识
        results = []
        for i in indices:
            results.append(self.knowledge[i])
            
        return results 