import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        # 使用预训练的ResNet作为基础模型
        resnet = models.resnet50(pretrained=True)
        # 移除最后的全连接层
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # 添加投影层，将特征映射到指定维度
        self.fc = nn.Linear(2048, embedding_dim)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features
    
    def preprocess(self, image_path):
        """预处理图像"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        return image 