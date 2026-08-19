import os
import json
import torch
import argparse
import time
import struct
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from model import PassDNVAE

def decode_line(raw):
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("cp949", errors="ignore").strip()

def build_vocab_if_needed(txt_path, vocab_path, min_occ):
    if os.path.exists(vocab_path):
        print(f"[INFO] Vocab already exists: {vocab_path}")
        return

    print(f"[INFO] Building vocab from: {txt_path}")

    counter = defaultdict(int)

    with open(txt_path, "rb") as f:
        for raw in f:
            line = decode_line(raw)
            if not line:
                continue

            for ch in line:
                counter[ch] += 1

    w2i = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    i2w = {
        0: "<pad>",
        1: "<sos>",
        2: "<eos>",
        3: "<unk>",
    }

    for ch, cnt in counter.items():
        if cnt >= min_occ:
            idx = len(w2i)
            w2i[ch] = idx
            i2w[idx] = ch

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {"w2i": w2i, "i2w": i2w},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[INFO] Vocab saved: {vocab_path}")
    print(f"[INFO] Vocab size: {len(w2i)}")

def build_offset_if_needed(txt_path, offset_path):
    if os.path.exists(offset_path):
        print(f"[INFO] Offset already exists: {offset_path}")
        return

    print(f"[INFO] Building offset file from: {txt_path}")

    n = 0

    with open(txt_path, "rb") as f, open(offset_path, "wb") as out:
        while True:
            pos = f.tell()
            raw = f.readline()

            if not raw:
                break

            line = decode_line(raw)
            if not line:
                continue

            out.write(struct.pack("<Q", pos))
            n += 1

    print(f"[INFO] Offset saved: {offset_path}")
    print(f"[INFO] Train samples: {n:,}")

class OffsetTextDataset(IterableDataset):

    def __init__(self, txt_path, vocab_path, offset_path, max_len, base_seed=1234):
        super().__init__()

        self.txt_path = txt_path
        self.offset_path = offset_path
        self.max_len = max_len
        self.base_seed = base_seed
        self.epoch = 0

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.w2i = vocab["w2i"]

        self.pad_idx = self.w2i["<pad>"]
        self.sos_idx = self.w2i["<sos>"]
        self.eos_idx = self.w2i["<eos>"]
        self.unk_idx = self.w2i["<unk>"]

        self.num_samples = os.path.getsize(offset_path) // 8

        self.offsets = np.fromfile(offset_path, dtype="<u8")

        if len(self.offsets) != self.num_samples:
            raise RuntimeError(
                f"Offset count mismatch: "
                f"expected={self.num_samples}, actual={len(self.offsets)}"
            )

        print(f"[INFO] Train samples: {self.num_samples:,}")
        print(
            f"[INFO] Offset memory usage: "
            f"{self.offsets.nbytes / (1024 ** 3):.3f} GB"
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _encode_line(self, line):
        ids = [self.w2i.get(ch, self.unk_idx) for ch in line]
        ids = [self.sos_idx] + ids + [self.eos_idx]

        length = min(len(ids) - 1, self.max_len)

        input_ids = ids[:length] + [self.pad_idx] * (self.max_len - length)
        target_ids = ids[1:length + 1] + [self.pad_idx] * (self.max_len - length)

        return {
            "input": torch.tensor(input_ids, dtype=torch.long),
            "target": torch.tensor(target_ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        seed = self.base_seed + self.epoch

        offsets = self.offsets.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(offsets)

        offsets = offsets[worker_id::num_workers]

        with open(self.txt_path, "rb") as txt_file:
            for offset in offsets:
                txt_file.seek(int(offset))
                raw = txt_file.readline()
                line = decode_line(raw)

                if not line:
                    continue

                yield self._encode_line(line)

def loss_function(logp, target, length, mean, logv, pad_idx):
    max_len = torch.max(length).item()

    logp = logp[:, :max_len].contiguous().view(-1, logp.size(-1))
    target = target[:, :max_len].contiguous().view(-1)

    nll = torch.nn.NLLLoss(ignore_index=pad_idx, reduction="sum")(logp, target)
    kl = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

    return nll + kl, nll.item(), kl.item()

def infinite_loader(loader, dataset):
    epoch = 0

    while True:
        dataset.set_epoch(epoch)

        for batch in loader:
            yield batch

        epoch += 1

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    train_txt_path = os.path.join(args.data_dir, args.train_file)
    train_stem = os.path.splitext(os.path.basename(args.train_file))[0]
    dataset_name = train_stem[:-6] if train_stem.endswith("-train") else train_stem
    vocab_path = os.path.join(args.data_dir, f"{dataset_name}-vocab.json") # Vocabulary filename setting
    offset_path = os.path.join(args.data_dir, f"{train_stem}.offset.bin")  # Offset filename setting

    build_vocab_if_needed(
        txt_path=train_txt_path,
        vocab_path=vocab_path,
        min_occ=args.min_occ,
    )

    build_offset_if_needed(
        txt_path=train_txt_path,
        offset_path=offset_path,
    )

    train_set = OffsetTextDataset(
        txt_path=train_txt_path,
        vocab_path=vocab_path,
        offset_path=offset_path,
        max_len=args.max_sequence_length,
        base_seed=args.base_seed,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = PassDNVAE(
        vocab_size=len(train_set.w2i),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        sos_idx=train_set.sos_idx,
        eos_idx=train_set.eos_idx,
        pad_idx=train_set.pad_idx,
        max_sequence_length=args.max_sequence_length,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("====================================")
    print("Model architecture:\n", model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Train samples: {len(train_set):,}")
    print("====================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_gen = infinite_loader(train_loader, train_set)

    global_step = 0
    total_nll = 0.0
    total_kl = 0.0
    best_loss = float("inf")

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "passdnvae_best.pt")

    model.train()

    while global_step < args.total_iters:
        batch = next(train_gen)

        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        length = batch["length"].to(device, non_blocking=True)

        logp, mean, logv, _ = model(x, length)
        loss, nll, kl = loss_function(logp, y, length, mean, logv, model.pad_idx)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_nll += nll
        total_kl += kl
        global_step += 1

        if global_step % args.log_every == 0:
            avg_nll = total_nll / (args.log_every * args.batch_size)
            avg_kl = total_kl / (args.log_every * args.batch_size)
            avg_total = avg_nll + avg_kl

            print(
                f"[Iter {global_step}/{args.total_iters}] "
                f"NLL={avg_nll:.4f} "
                f"KL={avg_kl:.4f} "
                f"TOTAL={avg_total:.4f} "
                f"time={(time.time() - start_time):.2f}s"
            )

            if avg_total < best_loss:
                best_loss = avg_total
                torch.save(model.state_dict(), best_path)
                print(f"[BEST @ {global_step}] loss={best_loss:.4f} -> {best_path}")

            if torch.cuda.is_available():
                print(
                    f"[GPU] "
                    f"allocated={torch.cuda.memory_allocated() / 1024**2:.2f}MB "
                    f"reserved={torch.cuda.memory_reserved() / 1024**2:.2f}MB"
                )

            total_nll = 0.0
            total_kl = 0.0

        if global_step % args.save_every == 0:
            save_path = os.path.join(
                args.save_dir,
                f"passdnvae_iter{global_step}.pt",
            )
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE @ {global_step}] {save_path}")

    final_path = os.path.join(args.save_dir, "passdnvae_final.pt")  
    torch.save(model.state_dict(), final_path)

    print("====================================")
    print(f"Training finished in {(time.time() - start_time):.2f}s")
    print(f"Final step: {global_step}")
    print(f"Final model saved: {final_path}")
    print("====================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="/path/dictionary") # Training dataset folder path
    parser.add_argument("--train_file", default="rockyou-train.txt") # Training dataset filename
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--word_dropout", type=float, default=0.0)
    parser.add_argument("--embedding_dropout", type=float, default=0.1)
    parser.add_argument("--max_sequence_length", type=int, default=12) # Maximum password length setting
    parser.add_argument("--min_occ", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    parser.add_argument("--total_iters", type=int, default=200000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--save_dir", default="/path/dictionary") # Model save path

    args = parser.parse_args()
    main(args)
