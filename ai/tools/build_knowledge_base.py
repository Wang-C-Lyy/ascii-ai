import json
import argparse
import os

def create_knowledge_base(input_dir, output_file):
    """从文本文件创建知识库"""
    knowledge_base = []
    
    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt') or filename.endswith('.md'):
            file_path = os.path.join(input_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 简单的QA对提取，假设格式为"Q: ... A: ..."
                qa_pairs = []
                lines = content.split('\n')
                i = 0
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    if line.startswith('Q:') or line.startswith('问:'):
                        question = line[2:].strip()
                        answer = ""
                        i += 1
                        
                        # 收集回答，直到下一个问题或文件结束
                        while i < len(lines) and not (lines[i].strip().startswith('Q:') or lines[i].strip().startswith('问:')):
                            if lines[i].strip().startswith('A:') or lines[i].strip().startswith('答:'):
                                answer = lines[i].strip()[2:].strip()
                            else:
                                answer += lines[i].strip() + " "
                            i += 1
                        
                        if question and answer:
                            qa_pairs.append({
                                'question': question,
                                'answer': answer.strip(),
                                'source': filename
                            })
                    else:
                        i += 1
                
                knowledge_base.extend(qa_pairs)
    
    # 保存知识库
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
    
    print(f"已创建知识库，包含 {len(knowledge_base)} 个问答对")

def main():
    parser = argparse.ArgumentParser(description="从文本文件创建知识库")
    parser.add_argument("--input_dir", type=str, required=True, help="输入文本文件目录")
    parser.add_argument("--output_file", type=str, required=True, help="输出知识库文件")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"错误：输入目录 {args.input_dir} 不存在")
        return
    
    create_knowledge_base(args.input_dir, args.output_file)

if __name__ == "__main__":
    main() 