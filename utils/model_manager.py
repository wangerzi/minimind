import os
import json
import torch
import glob
import gc
from datetime import datetime
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


class ModelManager:
    """通用的模型管理类，用于加载、保存和管理MiniMind模型"""
    
    def __init__(self):
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def unload_model(self):
        """卸载当前模型释放内存"""
        if self.current_model is not None:
            del self.current_model
            del self.current_tokenizer
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_path = None
            
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def load_pretrain_model(self, model_path, hidden_size=512, num_hidden_layers=8, use_moe=False):
        """加载预训练模型"""
        try:
            # 先卸载当前模型
            self.unload_model()
            
            # 加载tokenizer
            tokenizer = AutoTokenizer.from_pretrained('./model/')
            
            # 创建模型配置
            config = MiniMindConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                use_moe=use_moe
            )
            
            # 创建模型
            model = MiniMindForCausalLM(config)
            
            # 加载权重
            model.load_state_dict(torch.load(model_path, map_location=self.device), strict=True)
            model = model.eval().to(self.device)
            
            # 保存到实例变量
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_path = model_path
            
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            return True, f"模型加载成功！参数量: {param_count:.2f}M"
            
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"
    
    def load_sft_model(self, model_path, hidden_size=512, num_hidden_layers=8, use_moe=False):
        """加载SFT模型"""
        return self.load_pretrain_model(model_path, hidden_size, num_hidden_layers, use_moe)
    
    def generate_text_pretrain(self, prompt, max_length=512, temperature=0.85, top_p=0.85):
        """预训练模型文本生成"""
        if self.current_model is None or self.current_tokenizer is None:
            return "请先加载模型"
        
        try:
            tokenizer = self.current_tokenizer
            model = self.current_model
            
            # 预训练模型直接使用prompt，不需要chat template
            input_text = tokenizer.bos_token + prompt if tokenizer.bos_token else prompt
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=top_p,
                    temperature=temperature
                )
            
            # 只返回新生成的部分
            response = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            return response.strip()
            
        except Exception as e:
            return f"生成文本时出错: {str(e)}"
    
    def generate_text_sft(self, prompt, system_prompt="you are a helpful assistant", 
                         max_length=512, temperature=0.85, top_p=0.85):
        """SFT模型文本生成"""
        if self.current_model is None or self.current_tokenizer is None:
            return "请先加载模型"
        
        try:
            tokenizer = self.current_tokenizer
            model = self.current_model
            
            # 构建对话格式
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # 使用chat template
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=top_p,
                    temperature=temperature
                )
            
            # 只返回新生成的部分
            response = tokenizer.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            return response.strip()
            
        except Exception as e:
            return f"生成文本时出错: {str(e)}"
    
    def get_available_pretrain_models(self, out_dir="./out"):
        """获取可用的预训练模型列表"""
        if not os.path.exists(out_dir):
            return []
        
        model_files = glob.glob(os.path.join(out_dir, "pretrain_*.pth"))
        return [os.path.basename(f) for f in model_files]
    
    def get_available_sft_models(self, out_dir="./out"):
        """获取可用的SFT模型列表"""
        sft_dir = os.path.join(out_dir, "sft")
        if not os.path.exists(sft_dir):
            return []
        
        model_files = glob.glob(os.path.join(sft_dir, "*.pth"))
        return [os.path.basename(f) for f in model_files]
    
    def save_evaluation_results(self, results, model_name, model_type="pretrain", note=""):
        """保存评估结果到JSON文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eval_{model_name}_{timestamp}.json"
        
        eval_dir = f"eval_logs/{model_type}s"
        filepath = os.path.join(eval_dir, filename)
        
        eval_data = {
            "timestamp": timestamp,
            "model_name": model_name,
            "model_path": self.current_model_path,
            "model_type": model_type,
            "note": note,
            "results": results
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        
        return filepath
    
    def load_evaluation_history(self, model_type="pretrain"):
        """加载历史评估结果"""
        history_dir = f"eval_logs/{model_type}s"
        if not os.path.exists(history_dir):
            return []
        
        history_files = glob.glob(os.path.join(history_dir, "eval_*.json"))
        history_data = []
        
        for file_path in sorted(history_files, reverse=True):  # 按时间倒序
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history_data.append({
                        "filename": os.path.basename(file_path),
                        "timestamp": data.get("timestamp", ""),
                        "model_name": data.get("model_name", ""),
                        "note": data.get("note", ""),
                        "num_tests": len(data.get("results", [])),
                        "data": data,
                        "file_path": file_path
                    })
            except Exception as e:
                print(f"加载历史文件 {file_path} 失败: {e}")
        
        return history_data


# 全局模型管理器实例
model_manager = ModelManager()