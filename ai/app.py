from flask import Flask, request, jsonify, render_template
import torch
import os
from PIL import Image
import io
import base64

from models.vision_model import ImageEncoder
from models.language_model import TextEncoder
from models.fusion_model import MultimodalAgent
from utils.knowledge_base import KnowledgeBase
from config import config

app = Flask(__name__)

# 加载模型
def load_model():
    try:
        # 查找最新的模型检查点
        checkpoint_dir = config["model_save_path"]
        if not os.path.exists(checkpoint_dir):
            print(f"警告：检查点目录 {checkpoint_dir} 不存在，使用演示模式")
            demo_mode = True
        else:
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
            if not checkpoint_files:
                print("警告：未找到模型检查点，使用演示模式")
                demo_mode = True
            else:
                # 选择最新的检查点
                latest_checkpoint = sorted(checkpoint_files)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                demo_mode = False
        
        # 初始化模型
        vision_model = ImageEncoder(embedding_dim=config["vision_embedding_dim"])
        language_model = TextEncoder(embedding_dim=config["language_embedding_dim"])
        model = MultimodalAgent(
            vision_model=vision_model,
            language_model=language_model,
            hidden_dim=config["hidden_dim"],
            num_classes=config["num_classes"]
        )
        
        if not demo_mode:
            model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config["device"])))
            print(f"已加载模型检查点: {checkpoint_path}")
        else:
            print("使用未训练的模型（演示模式）")
        
        model = model.to(torch.device(config["device"]))
        model.eval()
        
        # 加载知识库（如果存在）
        knowledge_base = None
        knowledge_file = "data/knowledge_base.json"
        if os.path.exists(knowledge_file):
            try:
                knowledge_base = KnowledgeBase(knowledge_file)
                print("已加载知识库")
            except Exception as e:
                print(f"加载知识库失败: {e}")
        
        return model, knowledge_base, demo_mode
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None, True

model, knowledge_base, demo_mode = load_model()

@app.route("/")
def index():
    return render_template("index.html", demo_mode=demo_mode)

@app.route("/api/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "缺少图像"}), 400
    
    image_file = request.files["image"]
    question = request.form.get("question", "")
    
    # 保存临时图像
    temp_image_path = "temp_image.jpg"
    image_file.save(temp_image_path)
    
    try:
        # 处理图像
        image = model.vision_model.preprocess(temp_image_path)
        image = image.to(torch.device(config["device"]))
        
        # 图像分类
        with torch.no_grad():
            outputs = model(image, mode="vision_only")
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # 如果有问题，回答问题
        answer = None
        if question:
            tokens = model.language_model.encode_text(question)
            tokens = {k: v.to(torch.device(config["device"])) for k, v in tokens.items()}
            
            if demo_mode:
                answer = "演示模式：这是一个示例回答。实际应用中需要训练模型并可能接入生成式模型。"
            else:
                # 获取融合特征
                qa_features = model(image, tokens["input_ids"], tokens["attention_mask"], mode="qa")
                
                # 如果有知识库，使用知识库增强回答
                if knowledge_base:
                    question_text = model.language_model.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
                    relevant_knowledge = knowledge_base.search(question_text, k=3)
                    
                    # 简单回答生成策略
                    if relevant_knowledge:
                        answer = f"基于图像内容和相关知识，我认为答案是: {relevant_knowledge[0]['answer']}"
                    else:
                        answer = "我没有找到与这个问题相关的知识。"
                else:
                    answer = "我需要更多的专业知识来回答这个问题。"
        
        # 删除临时图像
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        
        # 返回结果
        result = {
            "class_id": pred_class,
            "confidence": confidence,
        }
        
        if answer:
            result["answer"] = answer
            
        return jsonify(result)
    
    except Exception as e:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return jsonify({"error": f"处理失败: {str(e)}"}), 500

if __name__ == "__main__":
    # 确保templates目录存在
    if not os.path.exists("templates"):
        os.makedirs("templates")
    app.run(debug=True)
