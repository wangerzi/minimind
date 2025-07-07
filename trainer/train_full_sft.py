import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    # 检查是否在分布式环境中
    try:
        if int(os.environ.get("RANK", -1)) != -1:  # DDP环境
            if dist.get_rank() == 0:
                print(content)
        else:  # 单GPU或CPU环境
            print(content)
    except:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def save_model_checkpoint(model, save_path, half_precision=True):
    """保存模型检查点"""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    if half_precision:
        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state_dict, save_path)
    Logger(f"模型已保存到: {save_path}")


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if args.save_interval > 0 and (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            ckp = os.path.join(args.sft_save_dir, f"{args.sft_model_name}_step_{epoch}_{step}.pth")
            save_model_checkpoint(model, ckp)
            model.train()

    # 每个epoch结束后保存检查点
    if not ddp or dist.get_rank() == 0:
        model.eval()
        epoch_ckp = os.path.join(args.sft_save_dir, f"{args.sft_model_name}.ep{epoch + 1}.pth")
        save_model_checkpoint(model, epoch_ckp)
        Logger(f"Epoch {epoch + 1} 训练完成，模型已保存")
        model.train()


def init_model(lm_config):
    """初始化模型，从指定的预训练模型加载"""
    tokenizer = AutoTokenizer.from_pretrained('../model')
    model = MiniMindForCausalLM(lm_config)
    
    # 加载预训练模型
    if args.pretrain_model_name:
        pretrain_path = os.path.join(args.out_dir, args.pretrain_model_name)
        if not os.path.exists(pretrain_path):
            raise FileNotFoundError(f"预训练模型文件不存在: {pretrain_path}")
        
        Logger(f"从预训练模型加载权重: {pretrain_path}")
        state_dict = torch.load(pretrain_path, map_location=args.device)
        model.load_state_dict(state_dict, strict=False)
    else:
        Logger("警告: 未指定预训练模型，将从随机初始化开始训练")

    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind SFT Training")
    
    # 模型相关参数
    parser.add_argument("--pretrain_model_name", type=str, required=True, 
                       help="预训练模型文件名 (如: pretrain_512.pth)")
    parser.add_argument("--sft_model_name", type=str, required=True,
                       help="SFT输出模型名称 (如: sft_512)")
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    
    # 数据相关参数
    parser.add_argument("--data_paths", type=str, nargs='+', required=True,
                       help="SFT训练数据文件路径列表")
    parser.add_argument("--shuffle_data", action="store_true", default=True,
                       help="是否打乱数据")
    
    # 训练相关参数
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=1)
    
    # 分布式训练参数
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument('--local_rank', type=int, default=-1)
    
    # Wandb相关参数
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SFT")

    args = parser.parse_args()

    # 配置路径
    args.sft_save_dir = os.path.join(args.out_dir, "sft")
    os.makedirs(args.sft_save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 验证数据文件存在性
    for data_path in args.data_paths:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    Logger(f"使用数据文件: {args.data_paths}")
    Logger(f"预训练模型: {args.pretrain_model_name}")
    Logger(f"SFT模型输出名称: {args.sft_model_name}")
    Logger(f"SFT模型保存目录: {args.sft_save_dir}")

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    )
    
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-SFT-{args.sft_model_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)

    # 创建SFT数据集，支持多个数据文件
    train_ds = SFTDataset(
        data_paths=args.data_paths, 
        tokenizer=tokenizer, 
        max_length=args.max_seq_len,
        shuffle=args.shuffle_data
    )
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,  # 数据集内部已经处理了shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    Logger(f"开始SFT训练，共 {args.epochs} 个epoch，每个epoch {iter_per_epoch} 步")
    
    for epoch in range(args.epochs):
        Logger(f"\n开始训练 Epoch {epoch + 1}/{args.epochs}")
        train_epoch(epoch, wandb)

    # 训练完成后保存最终模型
    if not ddp or dist.get_rank() == 0:
        model.eval()
        final_model_path = os.path.join(args.sft_save_dir, f"{args.sft_model_name}.pth")
        save_model_checkpoint(model, final_model_path)
        Logger(f"训练完成！最终模型已保存到: {final_model_path}")
