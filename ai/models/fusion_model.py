import torch
import torch.nn as nn

class MultimodalAgent(nn.Module):
    def __init__(self, vision_model, language_model, hidden_dim=512, num_classes=1000):
        super(MultimodalAgent, self).__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 分类头 - 用于图像识别任务
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # 问答头 - 用于专业问题回答
        self.qa_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 768)  # 映射回BERT维度用于生成回答
        )
        
    def forward(self, images, input_ids=None, attention_mask=None, mode="fusion"):
        # 图像编码
        image_features = self.vision_model(images)
        
        if mode == "vision_only":
            return self.classifier(image_features)
        
        # 文本编码
        text_features = self.language_model(input_ids, attention_mask)
        
        # 特征融合
        combined = torch.cat([image_features, text_features], dim=1)
        fused_features = self.fusion(combined)
        
        if mode == "classification":
            return self.classifier(fused_features)
        elif mode == "qa":
            return self.qa_head(fused_features)
        else:  # fusion
            return fused_features 