import streamlit as st
import os
import json
import torch
import glob
import pandas as pd
from datetime import datetime
import warnings
import random
import numpy as np
import sys
import gc

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

warnings.filterwarnings('ignore')

# 配置页面
st.set_page_config(
    page_title="MiniMind预训练模型评估工具",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 全局变量存储当前加载的模型
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
    st.session_state.current_tokenizer = None
    st.session_state.current_model_path = None

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_available_pretrain_models():
    """获取可用的预训练模型列表"""
    out_dir = "./out"
    if not os.path.exists(out_dir):
        return []
    
    model_files = glob.glob(os.path.join(out_dir, "pretrain_*.pth"))
    return [os.path.basename(f) for f in model_files]

def unload_current_model():
    """卸载当前模型释放内存"""
    if st.session_state.current_model is not None:
        del st.session_state.current_model
        del st.session_state.current_tokenizer
        st.session_state.current_model = None
        st.session_state.current_tokenizer = None
        st.session_state.current_model_path = None
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def load_model(model_path, hidden_size=512, num_hidden_layers=8, use_moe=False):
    """加载指定的模型"""
    try:
        # 先卸载当前模型
        unload_current_model()
        
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model = model.eval().to(device)
        
        # 保存到session state
        st.session_state.current_model = model
        st.session_state.current_tokenizer = tokenizer
        st.session_state.current_model_path = model_path
        
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        return True, f"模型加载成功！参数量: {param_count:.2f}M"
        
    except Exception as e:
        return False, f"模型加载失败: {str(e)}"

def generate_text(prompt, max_length=512, temperature=0.85, top_p=0.85):
    """生成文本"""
    if st.session_state.current_model is None or st.session_state.current_tokenizer is None:
        return "请先选择并加载模型"
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = st.session_state.current_tokenizer
        model = st.session_state.current_model
        
        # 预训练模型直接使用prompt，不需要chat template
        input_text = tokenizer.bos_token + prompt if tokenizer.bos_token else prompt
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True
        ).to(device)
        
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

def get_default_prompts():
    """获取默认的测试prompt列表"""
    return [
        '马克思主义基本原理',
        '人类大脑的主要功能',
        '万有引力原理是',
        '世界上最高的山峰是',
        '二氧化碳在空气中',
        '地球上最大的动物有',
        '杭州市的美食有',
        '当实验数据与理论预测出现偏差时，科学家首先应该考虑的是',
        '在量子力学中，粒子的状态是由',
        '金融市场的黑天鹅事件往往具有三个共同特征：',
        '在生物学中，基因的突变会导致',
        '在经济学中，边际效用递减规律表明',
        '在物理学中，光的折射率与',
        '在化学中，化学反应的速率与',
        '区块链技术的不可篡改性主要依赖于',
        # 《道诡异仙》相关测试
        '道诡异仙的世界中',
        '李火旺在现实世界的妻子是',
        '袄景教的人修炼功法利用',
        '高智坚的真实身份是',
        '袄景教修炼功法的核心原理是',
        '高智坚的真实身份其实是',
        '坐忘道的修行方式是',
        '诸葛渊与李火旺的关系本质是',
        '大梁皇帝司命的秘密是',
        '白玉京的真相是',
        '心素的能力具体表现为',
        '巴虺的信仰者会',
        '天陈国的历史隐藏着',
        '清风观的吕秀才实际上是',
        "修真境界'坐忘'指的是",
        '李火旺看到的幻觉中经常出现',
        '兵家修士的修炼需要',
        '骰子在坐忘道中象征',
        '大傩的仪式必须包含',
        '《大千录》记载的禁忌包括',
        '法教信徒获得力量的方式是',
        '监天司的创立目的是',
        '心浊现象的典型特征是',
        '丹阳子成仙的代价是',
        '腊月十八事件中失踪的',
        '修真者对抗癫火的方法有',
        '大齐王朝覆灭的真正原因是',
        '季灾这个名字暗示了',
        '无生老母的预言中提到',
        '玄牝的来历与有关',
        '幽都的入口隐藏在',
    ]

def save_evaluation_results(results, model_name, note=""):
    """保存评估结果到JSON文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_{model_name}_{timestamp}.json"
    filepath = os.path.join("eval_logs/pretrains", filename)
    
    eval_data = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": st.session_state.current_model_path,
        "note": note,
        "results": results
    }
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    return filepath

def load_evaluation_history():
    """加载历史评估结果"""
    history_dir = "eval_logs/pretrains"
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
            st.error(f"加载历史文件 {file_path} 失败: {e}")
    
    return history_data

def main():
    st.title("🧠 MiniMind预训练模型评估工具")
    st.markdown("---")
    
    # 侧边栏：模型选择
    with st.sidebar:
        st.header("🔧 模型配置")
        
        # 获取可用模型
        available_models = get_available_pretrain_models()
        
        if not available_models:
            st.error("未找到预训练模型文件，请确保out目录下有pretrain_*.pth文件")
            return
        
        # 模型选择
        model_options = ["请选择模型"] + available_models
        selected_model = st.selectbox(
            "选择预训练模型",
            model_options,
            index=0
        )
        
        # 模型参数配置
        hidden_size = st.selectbox(
            "Hidden Size",
            [512, 640, 768],
            index=0
        )
        
        num_hidden_layers = st.selectbox(
            "Hidden Layers",
            [8, 16],
            index=0
        )
        
        use_moe = st.checkbox("使用MoE")
        
        # 当选择改变时加载模型
        if selected_model != "请选择模型":
            model_path = f"./out/{selected_model}"
            current_selection = f"{selected_model}_{hidden_size}_{num_hidden_layers}_{use_moe}"
            
            if (st.session_state.current_model_path != model_path or 
                st.session_state.get('current_config') != current_selection):
                
                with st.spinner("正在加载模型..."):
                    success, message = load_model(model_path, hidden_size, num_hidden_layers, use_moe)
                    if success:
                        st.success(message)
                        st.session_state.current_config = current_selection
                    else:
                        st.error(message)
        
        # 显示当前模型状态
        if st.session_state.current_model is not None:
            st.success("✅ 模型已加载")
            st.info(f"当前模型: {os.path.basename(st.session_state.current_model_path)}")
        else:
            st.warning("⚠️ 未加载模型")
    
    # 主界面
    if st.session_state.current_model is None:
        st.warning("请先在侧边栏选择并加载模型")
        return
    
    # 创建标签页
    tab1, tab2 = st.tabs(["🧪 模型测试", "📊 历史结果"])
    
    with tab1:
        st.header("模型测试")
        
        # 测试输入区域
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 默认prompt
            default_prompts = get_default_prompts()
            
            # 将默认prompts转换为文本格式
            default_text = '\n'.join(default_prompts)
            
            # 直接使用文本输入
            custom_input = st.text_area(
                "测试Prompts（每行一个prompt）",
                value=default_text,
                height=250,
                placeholder="请输入测试内容，每行一个prompt",
                help="可以直接编辑默认prompts，或者添加新的测试内容",
                key="prompt_input"
            )
            
            test_prompts = [line.strip() for line in custom_input.split('\n') if line.strip()]
        
        with col2:
            # 生成参数
            st.subheader("生成参数")
            max_length = st.slider("最大生成长度", 50, 1024, 256)
            temperature = st.slider("Temperature", 0.1, 2.0, 0.85)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.85)
            
            # 执行次数
            num_runs = st.number_input("执行次数", min_value=1, max_value=10, value=3)
            
            # 随机种子设置
            use_fixed_seed = st.checkbox("使用固定种子")
            if use_fixed_seed:
                fixed_seed = st.number_input("随机种子", value=2025)
            
            # 备注信息
            st.markdown("---")
            test_note = st.text_area(
                "测试备注",
                placeholder="为本次测试添加备注信息...",
                height=80,
                help="可以记录测试目的、参数调整原因等信息"
            )
        
        # 开始测试按钮
        if st.button("🚀 开始测试", type="primary", disabled=len(test_prompts) == 0):
            if not test_prompts:
                st.error("请先选择或输入测试内容")
                return
            
            # 执行测试
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            total_tests = len(test_prompts) * num_runs
            current_test = 0
            
            for run_idx in range(num_runs):
                run_results = []
                
                for prompt_idx, prompt in enumerate(test_prompts):
                    current_test += 1
                    progress = current_test / total_tests
                    progress_bar.progress(progress)
                    status_text.text(f"执行第 {run_idx + 1} 轮，测试 {prompt_idx + 1}/{len(test_prompts)}: {prompt[:30]}...")
                    
                    # 设置随机种子
                    if use_fixed_seed:
                        setup_seed(fixed_seed)
                    else:
                        setup_seed(random.randint(0, 2048))
                    
                    # 生成文本
                    response = generate_text(prompt, max_length, temperature, top_p)
                    
                    result = {
                        "run": run_idx + 1,
                        "prompt": prompt,
                        "response": response,
                        "timestamp": datetime.now().isoformat(),
                        "parameters": {
                            "max_length": max_length,
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    }
                    run_results.append(result)
                
                all_results.extend(run_results)
            
            progress_bar.progress(1.0)
            status_text.text("测试完成！")
            
            # 保存结果
            model_name = os.path.basename(st.session_state.current_model_path).replace('.pth', '')
            saved_file = save_evaluation_results(all_results, model_name, test_note)
            st.success(f"测试结果已保存到: {saved_file}")
            
            # # 显示结果
            # with results_container:
            #     st.subheader("测试结果")
            #     for i, result in enumerate(all_results):
            #         with st.expander(f"第{result['run']}轮 - {result['prompt'][:50]}..."):
            #             st.text_area(
            #                 f"输入 (第{result['run']}轮)",
            #                 result['prompt'],
            #                 height=60,
            #                 disabled=True
            #             )
            #             st.text_area(
            #                 "输出",
            #                 result['response'],
            #                 height=120,
            #                 disabled=True
            #             )
    
    with tab2:
        st.header("历史测试结果")
        
        # 加载历史数据
        history_data = load_evaluation_history()
        
        if not history_data:
            st.info("暂无历史测试结果")
            return
        
        # 选择历史文件
        history_options = [
            f"{item['timestamp']} - {item['model_name']} ({item['num_tests']}个测试)" + 
            (f" - {item['note'][:30]}..." if item['note'] and len(item['note']) > 30 else f" - {item['note']}" if item['note'] else "")
            for item in history_data
        ]
        
        selected_history = st.selectbox(
            "选择历史结果",
            range(len(history_options)),
            format_func=lambda x: history_options[x]
        )
        
        if selected_history is not None:
            selected_data = history_data[selected_history]['data']
            selected_file_path = history_data[selected_history]['file_path']
            
            # 显示概览信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("测试时间", selected_data['timestamp'])
            with col2:
                st.metric("模型名称", selected_data['model_name'])
            with col3:
                st.metric("测试数量", len(selected_data['results']))
            with col4:
                # 计算平均响应长度
                avg_length = np.mean([len(r['response']) for r in selected_data['results']])
                st.metric("平均响应长度", f"{avg_length:.1f}")
            
            # 备注信息显示和编辑
            st.markdown("---")
            st.subheader("测试备注")
            
            # 使用session state来管理编辑模式
            edit_key = f"edit_note_{selected_history}"
            if edit_key not in st.session_state:
                st.session_state[edit_key] = False
            
            current_note = selected_data.get('note', '')
            
            if not st.session_state[edit_key]:
                # 显示模式
                col_note1, col_note2 = st.columns([4, 1])
                with col_note1:
                    if current_note:
                        st.text_area("", value=current_note, height=80, disabled=True, key=f"note_display_{selected_history}")
                    else:
                        st.text("暂无备注信息")
                with col_note2:
                    if st.button("编辑备注", key=f"edit_btn_{selected_history}"):
                        st.session_state[edit_key] = True
                        st.rerun()
            else:
                # 编辑模式
                new_note = st.text_area(
                    "编辑备注",
                    value=current_note,
                    height=80,
                    key=f"note_edit_{selected_history}"
                )
                col_save1, col_save2, col_save3 = st.columns([1, 1, 3])
                with col_save1:
                    if st.button("保存", key=f"save_btn_{selected_history}"):
                        # 更新JSON文件
                        selected_data['note'] = new_note
                        with open(selected_file_path, 'w', encoding='utf-8') as f:
                            json.dump(selected_data, f, ensure_ascii=False, indent=2)
                        st.session_state[edit_key] = False
                        st.success("备注已更新")
                        st.rerun()
                with col_save2:
                    if st.button("取消", key=f"cancel_btn_{selected_history}"):
                        st.session_state[edit_key] = False
                        st.rerun()
            
            st.markdown("---")
            
            # 创建结果表格
            results_df = pd.DataFrame([
                {
                    "轮次": r['run'],
                    "输入": r['prompt'][:50] + "..." if len(r['prompt']) > 50 else r['prompt'],
                    "输出": r['response'][:100] + "..." if len(r['response']) > 100 else r['response'],
                    "输出长度": len(r['response']),
                    "时间": r['timestamp']
                }
                for r in selected_data['results']
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # 详细查看
            st.subheader("详细结果")
            for i, result in enumerate(selected_data['results']):
                with st.expander(f"第{result['run']}轮 - {result['prompt'][:50]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area(
                            "输入",
                            result['prompt'],
                            height=100,
                            disabled=True,
                            key=f"hist_input_{i}"
                        )
                    with col2:
                        st.text_area(
                            "输出", 
                            result['response'],
                            height=100,
                            disabled=True,
                            key=f"hist_output_{i}"
                        )

if __name__ == "__main__":
    main() 