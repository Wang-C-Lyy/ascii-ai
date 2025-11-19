from transformers import BertModel, BertTokenizer
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.projection = nn.Linear(768, embedding_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS]标记的输出作为文本表示
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(cls_output)
        return projected
    
    def encode_text(self, text):
        """编码文本"""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        return tokens 