#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成预训练数据集脚本：将novel.txt和问答数据转换为预训练格式
"""

import json
import re
import os
from transformers import AutoTokenizer


def process_novel_to_pretrain(input_file, output_file, tokenizer_path="../model/", max_length=512):
    """
    将小说文本处理为预训练数据格式
    
    Args:
        input_file: 输入的小说文件路径
        output_file: 输出的jsonl文件路径
        tokenizer_path: tokenizer路径
        max_length: 最大序列长度
    """
    print(f"开始处理小说文件 {input_file}...")
    
    # 加载tokenizer
    print(f"加载tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 读取小说文本
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理文本
    processed_lines = []
    chapter_pattern = re.compile(r'^第\d+章')
    
    for line in lines:
        # 去除左右空白
        line = line.strip()
        
        # 跳过空行
        if not line:
            continue
            
        # 忽略章节标题
        if chapter_pattern.match(line):
            print(f"忽略章节标题: {line}")
            continue
        
        # 在每行前后添加标记
        formatted_line = f"<|im_start|>{line}<|im_end|>"
        processed_lines.append(formatted_line)
    
    print(f"处理了 {len(processed_lines)} 行有效文本")
    
    # 拼接数据并按最大长度分组
    output_data = []
    current_text = ""
    current_tokens = 0
    
    for line in processed_lines:
        # 计算当前行的token数量
        line_tokens = len(tokenizer.encode(line, add_special_tokens=False))
        
        # 如果加上当前行会超出最大长度，则保存当前文本
        if current_tokens + line_tokens > max_length and current_text:
            output_data.append({"text": current_text.strip()})
            current_text = ""
            current_tokens = 0
        
        # 添加当前行到文本中
        if current_text:
            current_text += " " + line
        else:
            current_text = line
        current_tokens += line_tokens
    
    # 处理最后剩余的文本
    if current_text.strip():
        output_data.append({"text": current_text.strip()})
    
    # 写入输出文件
    print(f"生成了 {len(output_data)} 条训练数据")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据已保存到 {output_file}")
    
    # 统计信息
    total_tokens = sum(len(tokenizer.encode(item["text"], add_special_tokens=False)) for item in output_data)
    avg_tokens = total_tokens / len(output_data) if output_data else 0
    
    print(f"统计信息:")
    print(f"  总条数: {len(output_data)}")
    print(f"  总token数: {total_tokens}")
    print(f"  平均token数: {avg_tokens:.2f}")
    print(f"  最大长度限制: {max_length}")


def process_qa_to_pretrain(input_file, output_file, tokenizer_path="../model/", max_length=512):
    """
    将问答数据处理为预训练数据格式
    
    Args:
        input_file: 输入的问答json文件路径（JSON数组格式）
        output_file: 输出的jsonl文件路径
        tokenizer_path: tokenizer路径
        max_length: 最大序列长度
    """
    print(f"开始处理问答文件 {input_file}...")
    
    # 加载tokenizer
    print(f"加载tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 读取问答数据（JSON数组格式）
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    
    print(f"读取了 {len(qa_data)} 条问答数据")
    
    # 处理问答对话
    processed_texts = []
    
    for item in qa_data:
        conversations = item.get('conversations', [])
        if not conversations:
            continue
        
        # 拼接对话内容
        dialogue_text = ""
        # 对话的问题和结果需要直接拼在一起，保留 im_start 和 im_end 标识，不需要 role 标识
        # 例如：<|im_start|>human: 问题内容 回答内容<|im_end|>
        
        # 收集所有对话内容
        conversation_parts = []
        for conv in conversations:
            role = conv.get('from', '')
            content = conv.get('value', '')
            if content:
                conversation_parts.append(f"{content}")
        
        # 将所有对话内容放在一个标记对中
        if conversation_parts:
            dialogue_text = f"<|im_start|>{' '.join(conversation_parts)}<|im_end|>"
        
        if dialogue_text:
            processed_texts.append(dialogue_text)
    
    print(f"处理了 {len(processed_texts)} 条有效对话")
    
    # 按最大长度分组
    output_data = []
    current_text = ""
    current_tokens = 0
    
    for text in processed_texts:
        # 计算当前文本的token数量
        text_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        
        # 如果加上当前文本会超出最大长度，则保存当前文本
        if current_tokens + text_tokens > max_length and current_text:
            output_data.append({"text": current_text.strip()})
            current_text = ""
            current_tokens = 0
        
        # 添加当前文本
        if current_text:
            current_text += " " + text
        else:
            current_text = text
        current_tokens += text_tokens
    
    # 处理最后剩余的文本
    if current_text.strip():
        output_data.append({"text": current_text.strip()})
    
    # 写入输出文件
    print(f"生成了 {len(output_data)} 条训练数据")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"数据已保存到 {output_file}")
    
    # 统计信息
    total_tokens = sum(len(tokenizer.encode(item["text"], add_special_tokens=False)) for item in output_data)
    avg_tokens = total_tokens / len(output_data) if output_data else 0
    
    print(f"统计信息:")
    print(f"  总条数: {len(output_data)}")
    print(f"  总token数: {total_tokens}")
    print(f"  平均token数: {avg_tokens:.2f}")
    print(f"  最大长度限制: {max_length}")


def main():
    """主函数"""
    tokenizer_path = "../model/"
    max_length = 512
    
    # 处理小说文本
    novel_input = "novel.txt"
    novel_output = "pretrain_novel.jsonl"
    
    if os.path.exists(novel_input):
        print("=" * 50)
        print("处理小说文本数据")
        print("=" * 50)
        process_novel_to_pretrain(novel_input, novel_output, tokenizer_path, max_length)
    else:
        print(f"小说文件 {novel_input} 不存在，跳过小说处理")
    
    # 处理问答数据
    qa_input = "daoguiyixian-sharegpt-qa-v2.json"  # 实际的问答数据文件
    qa_output = "pretrain_qa_sft.jsonl"
    
    if os.path.exists(qa_input):
        print("=" * 50)
        print("处理问答数据")
        print("=" * 50)
        process_qa_to_pretrain(qa_input, qa_output, tokenizer_path, max_length)
    else:
        print(f"问答文件 {qa_input} 不存在，跳过问答处理")
        print(f"提示：请将问答数据保存为 {qa_input} 格式，JSON数组格式")
        print("示例格式:")
        print("""[{"conversations": [{"from": "human", "value": "问题内容"}, {"from": "gpt", "value": "回答内容"}]}]""")
    
    print("=" * 50)
    print("所有处理完成!")


if __name__ == "__main__":
    main() 