import argparse
import json
import math
import random
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

ProjectPath = Path(__file__).resolve().parent.parent.parent
DataPath = ProjectPath / 'data'
SplitData = DataPath / 'split_by_repo'
ModelPath = ProjectPath / 'model/qwen3.5-0.8b-Base'
OutputPath = ProjectPath / 'output/cpt'


class TrainLogger:
    """
    控制台输出简洁信息，同时把详细日志写入 txt 文件
    """

    def __init__(
            self,
            output_dir: Path
    ) -> None:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.LogDir = output_dir / 'logs' / timestamp
        self.LogDir.mkdir(parents=True, exist_ok=True)
        self.LogPath = self.LogDir / 'train.txt'

    def _Write(
            self,
            level: str,
            message: str
    ) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(self.LogPath, 'a', encoding='utf-8') as file:
            file.write(f'[{timestamp}] [{level}] {message}\n')

    def Info(
            self,
            console_message: str,
            detail_message: str | None = None
    ) -> None:
        print(console_message)
        self._Write('INFO', detail_message or console_message)

    def Detail(
            self,
            message: str
    ) -> None:
        self._Write('INFO', message)


class PackedSequenceDataset(Dataset):
    """
    保存 packing 后的定长 token 序列
    """

    def __init__(
            self,
            Sequences: list[list[int]]
    ) -> None:
        self.Sequences = Sequences

    def __len__(self) -> int:
        return len(self.Sequences)

    def __getitem__(
            self,
            index: int
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(self.Sequences[index], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
        }


def ParseArgs() -> argparse.Namespace:
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='continued pretraining for typst coder'
    )
    parser.add_argument(
        '--train_data_path',
        type=Path,
        default=SplitData / 'train_repo/typst.json',
    )
    parser.add_argument(
        '--dev_data_path',
        type=Path,
        default=SplitData / 'dev_repo/typst.json',
    )
    parser.add_argument(
        '--model_path',
        type=Path,
        default=ModelPath,
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=OutputPath,
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=2048,
    )
    parser.add_argument(
        '--num_train_epochs',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.01,
    )
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.03,
    )
    parser.add_argument(
        '--logging_steps',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--save_steps',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--save_total_limit',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--max_train_docs',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--max_dev_docs',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
    )
    parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
    )

    return parser.parse_args()


def SetSeed(
        seed: int
) -> None:
    """
    固定随机种子，保证结果尽量可复现
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def LoadTypstData(
        data_path: Path,
        max_docs: int | None = None
) -> pd.DataFrame:
    """
    读取 typst 数据
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f'{data_path} 不存在，请先运行 src/data_process/split_by_repo.py'
        )

    Data = pd.read_json(data_path)

    if max_docs is not None:
        Data = Data.head(max_docs)

    return Data


def BuildPackedSequences(
        Data: pd.DataFrame,
        tokenizer,
        max_length: int
) -> list[list[int]]:
    """
    把多条文本编码并 packing 成固定长度序列

    这里的 packing 做法是：
    1. 每条文档单独 tokenize
    2. 在文档末尾追加 eos，作为显式边界
    3. 再把多条较短文档拼到同一个定长序列里
    4. 超长文档自然会被切成多个 chunk
    """
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError('tokenizer.eos_token_id 不能为空')
    if max_length <= 0:
        raise ValueError('max_length 必须大于 0')

    Sequences = []
    CurrentSequence = []

    for content in Data['content'].astype(str):
        token_ids = tokenizer.encode(
            content,
            add_special_tokens=False,
        )
        token_ids.append(eos_token_id)

        start = 0
        while start < len(token_ids):
            remain_length = max_length - len(CurrentSequence)
            CurrentSequence.extend(token_ids[start:start + remain_length])
            start += remain_length

            if len(CurrentSequence) == max_length:
                Sequences.append(CurrentSequence)
                CurrentSequence = []

    # 末尾剩余片段如果太短，训练价值较低，就直接丢弃
    if len(CurrentSequence) >= max(32, max_length // 4):
        pad_length = max_length - len(CurrentSequence)
        CurrentSequence.extend([eos_token_id] * pad_length)
        Sequences.append(CurrentSequence)

    return Sequences


def BuildDataLoader(
        Sequences: list[list[int]],
        batch_size: int,
        shuffle: bool
) -> DataLoader:
    """
    构造 DataLoader
    """
    DatasetObject = PackedSequenceDataset(Sequences)

    return DataLoader(
        DatasetObject,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def GetDevice() -> torch.device:
    """
    选择训练设备
    """
    if torch.cuda.is_available():
        return torch.device('cuda')

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def GetLoadDType(
        device: torch.device
) -> torch.dtype:
    """
    根据设备选择模型加载 dtype
    """
    if device.type == 'cuda':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        return torch.float16

    return torch.float32


def MoveBatchToDevice(
        Batch: dict[str, torch.Tensor],
        device: torch.device
) -> dict[str, torch.Tensor]:
    """
    把一个 batch 移动到目标设备
    """
    return {
        key: value.to(device)
        for key, value in Batch.items()
    }


def Evaluate(
        model,
        dataloader: DataLoader,
        device: torch.device
) -> dict[str, float]:
    """
    在 dev 集上计算平均 loss 和 perplexity
    """
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for Batch in dataloader:
            Batch = MoveBatchToDevice(Batch, device)
            Outputs = model(**Batch)

            total_loss += Outputs.loss.item()
            total_steps += 1

    average_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(average_loss) if average_loss < 20 else float('inf')

    return {
        'loss': average_loss,
        'perplexity': perplexity,
    }


def SaveCheckpoint(
        model,
        tokenizer,
        output_dir: Path,
        step: int,
        max_to_keep: int
) -> None:
    """
    保存 checkpoint，并删除过旧 checkpoint
    """
    checkpoint_dir = output_dir / f'checkpoint-{step}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    checkpoint_dirs = sorted(
        [
            path for path in output_dir.glob('checkpoint-*')
            if path.is_dir()
        ],
        key=lambda path: int(path.name.split('-')[-1])
    )

    while len(checkpoint_dirs) > max_to_keep:
        old_dir = checkpoint_dirs.pop(0)
        shutil.rmtree(old_dir)


def SaveTrainState(
        output_dir: Path,
        train_state: dict
) -> None:
    """
    保存训练状态到 json 文件
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'train_state.json', 'w', encoding='utf-8') as file:
        json.dump(train_state, file, ensure_ascii=False, indent=2)


def Train(
        args: argparse.Namespace
) -> None:
    """
    执行 continued pretraining
    """
    Logger = TrainLogger(args.output_dir)
    SetSeed(args.seed)

    device = GetDevice()
    dtype = GetLoadDType(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    TrainData = LoadTypstData(args.train_data_path, args.max_train_docs)
    DevData = LoadTypstData(args.dev_data_path, args.max_dev_docs)

    TrainSequences = BuildPackedSequences(TrainData, tokenizer, args.max_length)
    DevSequences = BuildPackedSequences(DevData, tokenizer, args.max_length)

    Logger.Info(
        console_message=f'log_file={Logger.LogPath}',
        detail_message=f'log_dir={Logger.LogDir}',
    )
    Logger.Info(
        console_message=(
            f'device={device} dtype={dtype} '
            f'train_docs={len(TrainData)} dev_docs={len(DevData)}'
        ),
        detail_message=(
            f'device={device} dtype={dtype} '
            f'train_docs={len(TrainData)} dev_docs={len(DevData)} '
            f'train_sequences={len(TrainSequences)} dev_sequences={len(DevSequences)}'
        ),
    )
    Logger.Detail(f'train_data_path={args.train_data_path}')
    Logger.Detail(f'dev_data_path={args.dev_data_path}')
    Logger.Detail(f'model_path={args.model_path}')
    Logger.Detail(f'output_dir={args.output_dir}')
    Logger.Detail(f'max_length={args.max_length}')
    Logger.Detail(f'num_train_epochs={args.num_train_epochs}')
    Logger.Detail(
        'batch_config='
        f'train_batch_size={args.per_device_train_batch_size}, '
        f'eval_batch_size={args.per_device_eval_batch_size}, '
        f'grad_accum={args.gradient_accumulation_steps}'
    )
    Logger.Detail(
        'optim_config='
        f'lr={args.learning_rate}, '
        f'weight_decay={args.weight_decay}, '
        f'warmup_ratio={args.warmup_ratio}'
    )

    if len(TrainSequences) == 0:
        raise ValueError('train_sequences 为空，请检查数据或减小 max_length')
    if len(DevSequences) == 0:
        raise ValueError('dev_sequences 为空，请检查数据或减小 max_length')

    if args.dry_run:
        Logger.Info(
            console_message='dry_run finished',
            detail_message='dry_run finished without starting training',
        )
        return

    TrainLoader = BuildDataLoader(
        TrainSequences,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
    )
    DevLoader = BuildDataLoader(
        DevSequences,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    )
    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_update_steps_per_epoch = math.ceil(
        len(TrainLoader) / args.gradient_accumulation_steps
    )
    max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
    warmup_steps = int(max_train_steps * args.warmup_ratio)

    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Logger.Detail(
        f'train_schedule='
        f'num_update_steps_per_epoch={num_update_steps_per_epoch}, '
        f'max_train_steps={max_train_steps}, '
        f'warmup_steps={warmup_steps}'
    )

    train_state = {
        'best_dev_loss': None,
        'best_checkpoint': None,
        'global_step': 0,
        'log_file': str(Logger.LogPath),
    }

    with open(args.output_dir / 'train_args.json', 'w', encoding='utf-8') as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2, default=str)

    global_step = 0
    accumulated_loss = 0.0

    for epoch in range(args.num_train_epochs):
        model.train()

        for step, Batch in enumerate(TrainLoader, start=1):
            Batch = MoveBatchToDevice(Batch, device)
            Outputs = model(**Batch)
            loss = Outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            accumulated_loss += Outputs.loss.item()

            if step % args.gradient_accumulation_steps == 0 or step == len(TrainLoader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    average_loss = accumulated_loss / args.logging_steps
                    learning_rate = scheduler.get_last_lr()[0]
                    Logger.Info(
                        console_message=(
                            f'train step={global_step} '
                            f'loss={average_loss:.4f}'
                        ),
                        detail_message=(
                            f'train epoch={epoch + 1} '
                            f'global_step={global_step} '
                            f'optimizer_step_loss={average_loss:.6f} '
                            f'learning_rate={learning_rate:.8f}'
                        ),
                    )
                    accumulated_loss = 0.0

                if global_step % args.eval_steps == 0:
                    Metrics = Evaluate(model, DevLoader, device)
                    Logger.Info(
                        console_message=(
                            f'eval step={global_step} '
                            f'loss={Metrics["loss"]:.4f} '
                            f'ppl={Metrics["perplexity"]:.4f}'
                        ),
                        detail_message=(
                            f'eval epoch={epoch + 1} '
                            f'global_step={global_step} '
                            f'dev_loss={Metrics["loss"]:.6f} '
                            f'dev_perplexity={Metrics["perplexity"]:.6f}'
                        ),
                    )

                    if (
                        train_state['best_dev_loss'] is None
                        or Metrics['loss'] < train_state['best_dev_loss']
                    ):
                        best_dir = args.output_dir / 'best_checkpoint'
                        best_dir.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(best_dir)
                        tokenizer.save_pretrained(best_dir)

                        train_state['best_dev_loss'] = Metrics['loss']
                        train_state['best_checkpoint'] = str(best_dir)
                        train_state['global_step'] = global_step
                        SaveTrainState(args.output_dir, train_state)
                        Logger.Detail(
                            f'best checkpoint updated: '
                            f'step={global_step}, '
                            f'best_dev_loss={Metrics["loss"]:.6f}, '
                            f'path={best_dir}'
                        )

                if global_step % args.save_steps == 0:
                    SaveCheckpoint(
                        model,
                        tokenizer,
                        args.output_dir,
                        global_step,
                        args.save_total_limit,
                    )
                    train_state['global_step'] = global_step
                    SaveTrainState(args.output_dir, train_state)
                    Logger.Detail(
                        f'checkpoint saved: '
                        f'step={global_step}, '
                        f'path={args.output_dir / f"checkpoint-{global_step}"}'
                    )

    FinalMetrics = Evaluate(model, DevLoader, device)
    Logger.Info(
        console_message=(
            f'final loss={FinalMetrics["loss"]:.4f} '
            f'ppl={FinalMetrics["perplexity"]:.4f}'
        ),
        detail_message=(
            f'final evaluation '
            f'dev_loss={FinalMetrics["loss"]:.6f} '
            f'dev_perplexity={FinalMetrics["perplexity"]:.6f}'
        ),
    )

    final_dir = args.output_dir / 'final_checkpoint'
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    Logger.Detail(f'final checkpoint saved: path={final_dir}')

    if (
        train_state['best_dev_loss'] is None
        or FinalMetrics['loss'] < train_state['best_dev_loss']
    ):
        train_state['best_dev_loss'] = FinalMetrics['loss']
        train_state['best_checkpoint'] = str(final_dir)

    train_state['global_step'] = global_step
    train_state['final_dev_loss'] = FinalMetrics['loss']
    train_state['final_dev_perplexity'] = FinalMetrics['perplexity']
    train_state['final_checkpoint'] = str(final_dir)
    SaveTrainState(args.output_dir, train_state)
    Logger.Info(
        console_message='training finished',
        detail_message=(
            f'training finished, '
            f'best_checkpoint={train_state["best_checkpoint"]}, '
            f'final_checkpoint={final_dir}'
        ),
    )


if __name__ == '__main__':
    Train(ParseArgs())
