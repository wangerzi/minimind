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
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


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
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0:
            save_model(model, lm_config, args, step=step)


def save_model(model, lm_config, args, epoch=None, step=None):
    """保存模型的通用函数
    
    Args:
        model: 要保存的模型
        lm_config: 模型配置
        args: 命令行参数
        epoch: 当前epoch（用于按epoch保存）
        step: 当前step（用于按step保存）
    """
    if not ddp or dist.get_rank() == 0:
        model.eval()
        moe_path = '_moe' if lm_config.use_moe else ''
        
        # 根据保存类型确定文件名
        if epoch is not None:
            # 按epoch保存
            base_name = args.output_model_name or f"pretrain_{lm_config.hidden_size}{moe_path}"
            ckp = f'{args.save_dir}/{base_name}_ep{epoch + 1}.pth'
        else:
            # 按step保存（原有逻辑）
            ckp = f'{args.save_dir}/{args.output_model_name or f"pretrain_{lm_config.hidden_size}{moe_path}"}.pth'

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        torch.save(state_dict, ckp)
        
        if epoch is not None:
            Logger(f'Epoch {epoch + 1} 模型已保存到: {ckp}')
        else:
            Logger(f'Step 检查点模型已保存到: {ckp}')
            
        model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    model = MiniMindForCausalLM(lm_config).to(args.device)
    
    # 如果指定了已有模型路径，则加载模型权重
    if args.load_model_path is not None:
        if os.path.exists(args.load_model_path):
            Logger(f'加载已有模型: {args.load_model_path}')
            state_dict = torch.load(args.load_model_path, map_location=args.device)
            # 处理可能的半精度模型权重
            state_dict = {k: v.float() if v.dtype == torch.float16 else v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            Logger('模型权重加载完成，继续预训练')
        else:
            Logger(f'警告: 指定的模型路径不存在 {args.load_model_path}，将从头开始训练')
    else:
        Logger('从头开始训练新模型')
    
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
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


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--num_attention_heads', default=8, type=int,
                        help="注意力头的数量")
    parser.add_argument('--num_key_value_heads', default=2, type=int,
                        help="键值对头的数量，用于分组查询注意力(GQA)")
    parser.add_argument("--data_path", type=str, nargs='+', default=["../dataset/pretrain_hq.jsonl"], 
                        help="数据路径，支持多个文件，例如: --data_path file1.jsonl file2.jsonl")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="指定已有模型路径进行继续训练，不指定则从头开始训练")
    parser.add_argument("--output_model_name", type=str, default=None,
                        help="指定输出模型的名称，不指定则使用默认名称")
    parser.add_argument("--shuffle_data", action="store_true", 
                        help="是否打乱训练数据，默认不打乱")
    parser.add_argument("--save_per_epoch", action="store_true",
                        help="是否在每个epoch结束时保存模型，文件名格式为 {model_name}_ep{epoch}.pth")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe
    )
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds, shuffle=args.shuffle_data) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=args.shuffle_data and not ddp,  # 只有在非分布式训练时才使用shuffle
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)

    # 打印数据打乱状态
    if args.shuffle_data:
        if ddp:
            Logger("数据打乱: 启用 (分布式训练模式)")
        else:
            Logger("数据打乱: 启用 (单机训练模式)")
    else:
        Logger("数据打乱: 禁用")

    for epoch in range(args.epochs):
        # 在分布式训练中，为每个epoch设置不同的随机种子
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)
        
        # 按epoch保存模型
        if args.save_per_epoch:
            save_model(model, lm_config, args, epoch=epoch)
