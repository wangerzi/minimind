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
from utils.model_manager import model_manager

warnings.filterwarnings('ignore')

# 配置页面
st.set_page_config(
    page_title="MiniMind SFT模型评估工具",
    page_icon="🤖",
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

def get_default_sft_prompts():
    """获取默认的SFT测试prompt列表"""
    return [
        '请介绍一下马克思主义基本原理',
        '解释一下人类大脑的主要功能',
        '万有引力原理是什么？',
        '世界上最高的山峰是哪一座？',
        '请告诉我杭州市有哪些著名美食',
        '当实验数据与理论预测出现偏差时，科学家首先应该考虑什么？',
        '在量子力学中，粒子的状态是由什么描述的？',
        '什么是金融市场的黑天鹅事件？',
        '在生物学中，基因突变会导致什么？',
        '请解释一下边际效用递减规律',
        '光的折射率与什么有关？',
        '化学反应的速率与哪些因素有关？',
        '区块链技术的不可篡改性主要依赖于什么？',
        '请帮我写一首关于春天的诗',
        '如何学习编程？',
        '请推荐几本好看的科幻小说',
        '怎样保持身体健康？',
        '请解释一下人工智能的发展历程',
        '如何提高工作效率？',
        '请介绍一下中国的传统节日',
        # 对话式问题
        '你是谁？你能做什么？',
        '请帮我制定一个学习计划',
        '我感到很焦虑，你能给我一些建议吗？',
        '请帮我分析一下这个问题的解决方案',
        '你觉得人工智能会取代人类吗？',
    ]

def main():
    st.title("🤖 MiniMind SFT模型评估工具")
    st.markdown("---")
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🔧 模型管理", "🧪 模型测试", "📊 历史结果"])
    
    with tab1:
        st.header("SFT模型管理")
        
        # 获取可用模型
        available_models = model_manager.get_available_sft_models()
        
        if not available_models:
            st.error("未找到SFT模型文件，请确保out/sft目录下有模型文件")
            st.info("💡 提示：运行SFT训练后，模型会保存在out/sft目录下")
        else:
            # 模型选择区域
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("选择SFT模型")
                
                # 模型选择
                model_options = ["请选择模型"] + available_models
                selected_model = st.selectbox(
                    "选择SFT模型",
                    model_options,
                    index=0
                )
                
                # 模型参数配置
                col_param1, col_param2, col_param3 = st.columns(3)
                with col_param1:
                    hidden_size = st.selectbox(
                        "Hidden Size",
                        [512, 640, 768],
                        index=0
                    )
                
                with col_param2:
                    num_hidden_layers = st.selectbox(
                        "Hidden Layers", 
                        [8, 16],
                        index=0
                    )
                
                with col_param3:
                    use_moe = st.checkbox("使用MoE")
                
                # 模型操作按钮
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    load_button = st.button("🔄 加载模型", type="primary", disabled=(selected_model == "请选择模型"))
                with col_btn2:
                    unload_button = st.button("🗑️ 卸载模型", disabled=(model_manager.current_model is None))
                
                # 处理加载模型
                if load_button and selected_model != "请选择模型":
                    model_path = f"./out/sft/{selected_model}"
                    with st.spinner("正在加载SFT模型..."):
                        success, message = model_manager.load_sft_model(model_path, hidden_size, num_hidden_layers, use_moe)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                
                # 处理卸载模型
                if unload_button:
                    model_manager.unload_model()
                    st.success("模型已卸载")
                    st.rerun()
            
            with col2:
                st.subheader("当前状态")
                # 显示当前模型状态
                if model_manager.current_model is not None:
                    st.success("✅ 模型已加载")
                    st.info(f"模型: {os.path.basename(model_manager.current_model_path)}")
                    
                    # 显示模型参数信息
                    param_count = sum(p.numel() for p in model_manager.current_model.parameters() if p.requires_grad) / 1e6
                    st.metric("参数量", f"{param_count:.2f}M")
                else:
                    st.warning("⚠️ 未加载模型")
                    st.info("请选择模型并点击加载按钮")
    
    with tab2:
        st.header("SFT模型测试")
        
        # 检查是否有加载的模型
        if model_manager.current_model is None:
            st.warning("⚠️ 请先在『模型管理』标签页中加载SFT模型")
            st.info("💡 提示：切换到『模型管理』标签页，选择模型并点击『加载模型』按钮")
        
        # 测试输入区域
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # System Prompt设置
            st.subheader("System Prompt")
            system_prompt = st.text_area(
                "设置System Prompt",
                value="you are a helpful assistant",
                height=100,
                help="设置模型的角色和行为指导"
            )
            
            # 默认prompt
            default_prompts = get_default_sft_prompts()
            
            # 将默认prompts转换为文本格式
            default_text = '\n'.join(default_prompts)
            
            # 直接使用文本输入
            custom_input = st.text_area(
                "测试Prompts（每行一个prompt）",
                value=default_text,
                height=250,
                placeholder="请输入测试内容，每行一个prompt",
                help="可以直接编辑默认prompts，或者添加新的测试内容",
                key="sft_prompt_input"
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
                        setup_seed(random.randint(0, 20480))
                    
                    # 生成文本
                    response = model_manager.generate_text_sft(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    result = {
                        "run": run_idx + 1,
                        "prompt": prompt,
                        "response": response,
                        "system_prompt": system_prompt,
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
            model_name = os.path.basename(model_manager.current_model_path).replace('.pth', '')
            saved_file = model_manager.save_evaluation_results(all_results, model_name, "sft", test_note)
            st.success(f"测试结果已保存到: {saved_file}")
    
    with tab3:
        st.header("历史测试结果")
        
        # 加载历史数据
        history_data = model_manager.load_evaluation_history("sft")
        
        if not history_data:
            st.info("📋 暂无历史测试结果")
            st.markdown("💡 **提示**: 在『模型测试』标签页中运行测试后，结果会显示在这里")
            return
        
        # 历史结果管理区域
        col_select, col_delete = st.columns([4, 1])
        
        with col_select:
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
        
        with col_delete:
            st.markdown("　")  # 占位符，对齐选择框
            delete_button = st.button("🗑️ 删除选中结果", type="secondary")
        
        # 处理删除操作
        if delete_button and selected_history is not None:
            selected_file_path = history_data[selected_history]['file_path']
            selected_filename = history_data[selected_history]['filename']
            
            os.remove(selected_file_path)
            st.success(f"✅ 已删除结果文件: {selected_filename}")
        
        # 显示选中的历史结果
        if selected_history is not None and not delete_button:
            selected_data = history_data[selected_history]['data']
            selected_file_path = history_data[selected_history]['file_path']
            
            st.markdown("---")
            
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
            
            # System Prompt显示
            if selected_data['results'] and 'system_prompt' in selected_data['results'][0]:
                st.subheader("System Prompt")
                st.code(selected_data['results'][0]['system_prompt'])
            
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
                            key=f"sft_hist_input_{i}"
                        )
                    with col2:
                        st.text_area(
                            "输出", 
                            result['response'],
                            height=100,
                            disabled=True,
                            key=f"sft_hist_output_{i}"
                        )

if __name__ == "__main__":
    main()