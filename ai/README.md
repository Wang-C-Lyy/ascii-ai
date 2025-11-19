# 多模态智能体

这是一个结合图像识别和专业问题回答能力的多模态智能体项目。该智能体能够处理图像输入和文本问题，提供图像分类结果和专业问题的回答。

## 功能特点

- **图像识别**：识别图像中的对象和场景
- **专业问题回答**：基于图像内容和知识库回答专业问题
- **多模态融合**：结合视觉和语言信息进行推理
- **知识库支持**：利用专业领域知识库增强回答质量

## 项目结构

```
multimodal_agent/
├── data/                   # 数据目录
│   ├── images/             # 图像数据
│   ├── text_corpus/        # 文本语料
│   ├── annotations.json    # 标注数据
│   └── knowledge_base.json # 知识库
├── models/                 # 模型目录
│   ├── vision_model.py     # 视觉模型
│   ├── language_model.py   # 语言模型
│   └── fusion_model.py     # 融合模型
├── utils/                  # 工具目录
│   ├── data_loader.py      # 数据加载器
│   └── knowledge_base.py   # 知识库工具
├── tools/                  # 工具脚本
│   ├── build_knowledge_base.py # 构建知识库
│   └── evaluate_model.py   # 评估模型
├── templates/              # Web应用模板
│   └── index.html          # 主页模板
├── checkpoints/            # 模型检查点
├── app.py                  # Web应用
├── train.py                # 训练脚本
├── inference.py            # 推理脚本
├── config.py               # 配置文件
└── requirements.txt        # 依赖列表
```

## 安装

1. 克隆仓库：
   ```
   git clone https://github.com/yourusername/multimodal_agent.git
   cd multimodal_agent
   ```

2. 创建并激活虚拟环境：
   ```
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. 安装依赖：
   ```
   pip install -r requirements.txt
   ```

## 数据准备

1. 将训练图像放在 `data/images/` 目录下
2. 准备标注文件 `data/annotations.json`，格式如下：
   ```json
   [
     {
       "image_filename": "image1.jpg",
       "question": "这是什么？",
       "answer": "这是一只猫。",
       "label": 0
     },
     ...
   ]
   ```

3. 构建知识库：
   ```
   python tools/build_knowledge_base.py --input_dir data/text_corpus --output_file data/knowledge_base.json
   ```

## 训练模型

```
python train.py
```

训练参数可以在 `config.py` 中修改。

## 推理

使用训练好的模型进行推理：

```
python inference.py --image path/to/image.jpg --question "这是什么？" --checkpoint checkpoints/model_epoch_30.pt --knowledge data/knowledge_base.json
```

## Web应用

启动Web应用：

```
python app.py
```

然后在浏览器中访问 http://localhost:5000

## 评估模型

```
python tools/evaluate_model.py --model_path checkpoints/model_epoch_30.pt --test_data data/test_annotations.json
```

## 许可证

MIT

## 联系方式


如有问题，请联系 yangs35301@outlook.com
