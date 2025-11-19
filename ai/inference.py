import torch
from PIL import Image
import argparse
import torch.nn.functional as F
import os

from models.vision_model import ImageEncoder
from models.language_model import TextEncoder
from models.fusion_model import MultimodalAgent
from utils.knowledge_base import KnowledgeBase
from config import config

def load_model(checkpoint_path):
    # 初始化模型
    vision_model = ImageEncoder(embedding_dim=config['vision_embedding_dim'])
    language_model = TextEncoder(embedding_dim=config['language_embedding_dim'])
    model = MultimodalAgent(
        vision_model=vision_model,
        language_model=language_model,
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes']
    )
    
    # 加载保存的权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config['device'])))
    model = model.to(torch.device(config['device']))
    model.eval()
    
    return model

def process_image(image_path, vision_model):
    """处理图像并返回特征"""
    image = vision_model.preprocess(image_path)
    image = image.to(torch.device(config['device']))
    return image

def process_text(question, language_model):
    """处理问题文本并返回编码"""
    tokens = language_model.encode_text(question)
    tokens = {k: v.to(torch.device(config['device'])) for k, v in tokens.items()}
    return tokens

def image_classification(model, image):
    """图像分类"""
    with torch.no_grad():
        outputs = model(image, mode="vision_only")
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    return pred_class, probs[0][pred_class].item()

def answer_question(model, image, tokens, knowledge_base=None):
    """回答关于图像的问题"""
    with torch.no_grad():
        # 获取融合特征
        qa_features = model(image, tokens['input_ids'], tokens['attention_mask'], mode="qa")
        
        # 如果有知识库，使用知识库增强回答
        if knowledge_base:
            question_text = model.language_model.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            relevant_knowledge = knowledge_base.search(question_text, k=3)
            
            # 构建上下文
            context = ""
            for item in relevant_knowledge:
                context += f"{item['question']}: {item['answer']}\n"
            
            # 简单回答生成策略（实际应用中可替换为更复杂的生成模型）
            answer = f"基于图像内容和相关知识，我认为答案是: {relevant_knowledge[0]['answer']}"
        else:
            # 简单回答（实际应用中需要接入生成式模型）
            answer = "这是一个示例回答，实际应用中需要接入生成式模型。"
        
    return answer

def main():
    parser = argparse.ArgumentParser(description="多模态智能体推理")
    parser.add_argument("--image", type=str, required=True, help="图像路径")
    parser.add_argument("--question", type=str, default="", help="问题文本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--knowledge", type=str, default="", help="知识库文件路径")
    args = parser.parse_args()
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"错误：图像文件 {args.image} 不存在")
        return
    
    # 检查检查点文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"错误：检查点文件 {args.checkpoint} 不存在")
        return
    
    # 加载模型
    try:
        model = load_model(args.checkpoint)
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 加载知识库（如果提供）
    knowledge_base = None
    if args.knowledge and os.path.exists(args.knowledge):
        try:
            knowledge_base = KnowledgeBase(args.knowledge)
            print("知识库加载成功")
        except Exception as e:
            print(f"加载知识库失败: {e}")
    
    # 处理图像
    image = process_image(args.image, model.vision_model)
    
    # 图像识别
    class_id, confidence = image_classification(model, image)
    print(f"图像分类结果: 类别 {class_id}, 置信度 {confidence:.4f}")
    
    # 如果提供了问题，回答问题
    if args.question:
        tokens = process_text(args.question, model.language_model)
        answer = answer_question(model, image, tokens, knowledge_base)
        print(f"问题: {args.question}")
        print(f"回答: {answer}")

if __name__ == "__main__":
    main() 