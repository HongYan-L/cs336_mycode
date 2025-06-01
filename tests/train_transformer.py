from .adapters import *
from tqdm import tqdm
import os
import wandb
import argparse
import torch
import time
import pickle
import pathlib
import numpy as np
DATA_PATH = (pathlib.Path(__file__).resolve().parent.parent) / "data"

def save_pkl(file, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)

def load_pkl(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
        return file

def save_encode(file, file_name):
    np.array(file, dtype=np.uint16).tofile(file_name)

def save_encode_stream(token_stream: Iterable[int], file_path: os.PathLike):
    array = np.fromiter(token_stream, dtype=np.uint16)
    array.tofile(file_path)

def train_bpe_TinyStories(
    file_name: str | os.PathLike, 
    vocab_size: int, 
    special_tokens: list[str], 
    vocab_name: str, 
    merges_name: str
):
    start_time = time.time()
    traindata_path = DATA_PATH / file_name
    vocab, merges = run_train_bpe(
        input_path=traindata_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    save_pkl(vocab, DATA_PATH / vocab_name)
    save_pkl(merges, DATA_PATH / merges_name)
    end_time = time.time()
    duration = end_time - start_time
    minutes = int(duration // 60)
    seconds = int(duration % 60)
    print(f"执行时间: {minutes} 分 {seconds} 秒")

def Tokenizer_TinyStories(
    trainfile_name: str | os.PathLike, 
    validfile_name: str | os.PathLike, 
    trainencode_name: str | os.PathLike, 
    validencode_name: str | os.PathLike, 
    vocab_name: str | os.PathLike, 
    merges_name: str | os.PathLike, 
    special_tokens: list[str]
):
    start_time = time.time()
    trainfile_path = DATA_PATH / trainfile_name
    validfile_path = DATA_PATH / validfile_name
    trainencode_path = DATA_PATH / trainencode_name
    validencode_path = DATA_PATH / validencode_name
    tokenizer = Tokenizer.from_files(DATA_PATH / vocab_name, DATA_PATH / merges_name, special_tokens)

    # 处理训练集（流式编码）
    with open(trainfile_path, 'r', encoding='utf-8') as f:
        train_lines = f.readlines()

    # total_bytes = sum(len(line.encode('utf-8')) for line in train_lines)
    # start_time = time.time()

    encode_stream = tokenizer.encode_iterable(train_lines)
    # token_list = list(encode_stream)
    # total_tokens = len(token_list)

    # 计算 tokenizer 压缩比
    # compression_ratio = total_bytes / total_tokens if total_tokens > 0 else float('inf')
    # print(f"Total bytes: {total_bytes}")
    # print(f"Total tokens: {total_tokens}")
    # print(f"Compression ratio (bytes/token): {compression_ratio:.4f}")

    save_encode_stream(encode_stream, trainencode_path)

    # end_time = time.time()
    # elapsed = end_time - start_time
    # 计算 tokenizer 的速度
    # throughput = total_bytes / elapsed / (1024 ** 2)  # MB/s
    # print(f"[Tokenizer Benchmark] Encoded {total_bytes / (1024 ** 3):.2f} GB in {elapsed:.2f}s")
    # print(f"[Tokenizer Benchmark] Throughput: {throughput:.2f} MB/s")

    # 处理验证集（流式编码）
    with open(validfile_path, 'r', encoding='utf-8') as f:
        valid_lines = f.readlines()
    encode_stream = tokenizer.encode_iterable(valid_lines)
    save_encode_stream(encode_stream, validencode_path)
    # with open(trainfile_path, 'r', encoding='utf-8') as f:
    #     traintext = f.read()
    # encode_traintext = tokenizer.encode(traintext)
    # save_encode(encode_traintext, trainencode_path)
    # with open(validfile_path, 'r', encoding='utf-8') as f:
    #     validtext = f.read()
    # encode_validtext = tokenizer.encode(validtext)
    # save_encode(encode_validtext, validencode_path)

def evaluate_validloss(model, valid_dataset, batch_size, context_length, device, num_batches=10):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for _ in range(num_batches):
            input_valid, target_valid = run_get_batch(valid_dataset, batch_size, context_length, device)
            logits = model(input_valid)
            loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_valid.view(-1))
            total_loss += loss.item()
    model.train()
    return total_loss / num_batches

# Argument parser
# def get_args():
#     parser = argparse.ArgumentParser()
    # parser.add_argument('--train_data', type=str, required=True)
    # parser.add_argument('--val_data', type=str, required=True)
    # parser.add_argument('--vocab_size', type=int, required=True)
    # parser.add_argument('--context_length', type=int, default=128)
    # parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--lr', type=float, default=3e-4)
    # parser.add_argument('--weight_decay', type=float, default=0.01)
    # parser.add_argument('--clip_norm', type=float, default=1.0)
    # parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--log_interval', type=int, default=100)
    # parser.add_argument('--ckpt_interval', type=int, default=500)
    # parser.add_argument('--ckpt', type=str, default='checkpoint.pt')
    # return parser.parse_args()

if __name__ == '__main__':
    trainfile_name = 'TinyStoriesV2-GPT4-train.txt'
    validfile_name = 'TinyStoriesV2-GPT4-valid.txt'
    vocab_name = 'TinyStories_vocab.pkl'
    merges_name = 'TinyStories_merges.pkl'
    trainencode_name = 'TStrain_tokens.bin'
    validencode_name = 'TSvalid_tokens.bin'
    vocab_size = 10000
    batch_size = 64
    context_length = 256
    d_model = 512
    d_ff = 1344
    rope_theta = 10000
    n_layers = 4
    n_heads = 16
    epochs = 10
    max_l2_norm = 1e-2
    log_interval = 200
    ckpt_interval = 500
    special_tokens = ["<|endoftext|>"]
    ckpt_name = 'TScheckpoint.pt'
    # train_bpe_TinyStories(trainfile_name, vocab_size, special_tokens, vocab_name, merges_name)
    # TinyStories_vocab = load_pkl(DATA_PATH / vocab_name)
    # TinyStories_merges = load_pkl(DATA_PATH / merges_name)
    # Tokenizer_TinyStories(trainfile_name, validfile_name, trainencode_name, validencode_name, vocab_name, merges_name, special_tokens)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")
    train_dataset = np.memmap(DATA_PATH / trainencode_name, dtype=np.uint16, mode="r")
    valid_dataset = np.memmap(DATA_PATH / validencode_name, dtype=np.uint16, mode="r")
    num_tokens = len(train_dataset)
    num_possible_sequences = num_tokens - context_length
    steps_per_epoch = num_possible_sequences // batch_size
    total_iters = steps_per_epoch * epochs
    print(f"Total iterations: {total_iters}")
    # init wandb
    wandb.init(
        project="transformer-lm",
        id="5s6xng5x",
        resume="must",
        name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
        config={
            "batch_size": batch_size,
            "context_length": context_length,
            "max_lr": 1e-3,
            "min_lr": 1e-4,
            "warmup_iters": 500,
            "cosine_iters": 10000,
        }
    )
    # model
    model = Transformer_lm(
        vocab_size=vocab_size, 
        context_length=context_length, 
        num_layers=n_layers, 
        d_model=d_model, 
        num_heads=n_heads, 
        d_ff=d_ff, 
        rope_theta=rope_theta
    ).to(device)
    # AdamW use default lr, betas, eps, weight_decay
    optimizer = AdamW(model.parameters())
    # Resume checkpoint
    start_iter = 0
    ckpt_path = DATA_PATH / ckpt_name
    if ckpt_path.exists():
        start_iter = run_load_checkpoint(src=ckpt_path, model=model, optimizer=optimizer)
    model.train()
    wandb.watch(model, log="all", log_freq=100)
    pbar = tqdm(total=total_iters)
    iteration = start_iter
    best_val_loss = float('inf')

    for epoch in range(epochs):
        for i in range(len(train_dataset) // (batch_size * context_length)):
            input_train, target_train = run_get_batch(train_dataset, batch_size, context_length, device)
            logits = model(input_train)
            loss = run_cross_entropy(logits.view(-1, logits.size(-1)), target_train.view(-1))
            lr = run_get_lr_cosine_schedule(
                iteration,
                max_learning_rate=1e-3,
                min_learning_rate=1e-4,
                warmup_iters=500,
                cosine_cycle_iters=10000,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.zero_grad()
            loss.backward()
            run_gradient_clipping(model.parameters(), max_l2_norm)
            optimizer.step()
            if iteration % log_interval == 0:
                print(f"[Iter {iteration}] loss: {loss.item():.4f}")
                wandb.log({"train/loss": loss.item(), "lr": lr, "iteration": iteration})
            if iteration % ckpt_interval == 0:
                run_save_checkpoint(model, optimizer, iteration, ckpt_path)
            iteration += 1
            pbar.update(1)
        # 每个 epoch 做验证
        val_loss = evaluate_validation_loss(model, valid_dataset, batch_size, context_length, device)
        print(f"[Epoch {epoch}] Validation loss: {val_loss:.4f}")
        wandb.log({"val/loss": val_loss, "epoch": epoch, "iteration": iteration})
        # 如果模型更好就保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            run_save_checkpoint(model, optimizer, iteration, "best_model.pt")
            print(f"Saved best model (val_loss={val_loss:.4f})")
            wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.finish()
