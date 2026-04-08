import os
import json
import torch
import argparse
import time
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from model import PassDNVAE


# ======================================================
# Utils
# ======================================================
def open_text_auto(path):
    try:
        return open(path, 'r', encoding='utf-8')
    except UnicodeDecodeError:
        return open(path, 'r', encoding='cp949')


# ======================================================
# Preprocess
# ======================================================
def preprocess_if_needed(data_dir, split, vocab_file, max_len, min_occ):
    txt_path = os.path.join(data_dir, f'rockyou-{split}.txt')    #File Name
    json_path = os.path.join(data_dir, f'rockyou-{split}.json')
    vocab_path = os.path.join(data_dir, vocab_file)

    if os.path.exists(json_path) and os.path.exists(vocab_path):
        return

    print(f"[INFO] Preprocessing {split.upper()}...")

    if split == 'train':
        with open_text_auto(txt_path) as f:
            lines = [list(line.strip()) for line in f]

        counter = defaultdict(int)
        for line in lines:
            for ch in line:
                counter[ch] += 1

        w2i = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
        i2w = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}

        for ch, cnt in counter.items():
            if cnt >= min_occ:
                idx = len(w2i)
                w2i[ch] = idx
                i2w[idx] = ch

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({'w2i': w2i, 'i2w': i2w}, f, ensure_ascii=False, indent=2)

    with open_text_auto(vocab_path) as f:
        w2i = json.load(f)['w2i']

    with open_text_auto(txt_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {}
    for i, line in enumerate(lines):
        ids = [w2i.get(ch, w2i['<unk>']) for ch in line]
        ids = [w2i['<sos>']] + ids + [w2i['<eos>']]
        length = min(len(ids) - 1, max_len)

        input_ids = ids[:length] + [w2i['<pad>']] * (max_len - length)
        target_ids = ids[1:length + 1] + [w2i['<pad>']] * (max_len - length)

        data[i] = {
            'input': input_ids,
            'target': target_ids,
            'length': length
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ======================================================
# Dataset
# ======================================================
class TextDataset(Dataset):
    def __init__(self, json_path, vocab_path):
        with open_text_auto(json_path) as f:
            self.data = json.load(f)
        with open_text_auto(vocab_path) as f:
            vocab = json.load(f)
        self.w2i = vocab['w2i']
        self.pad_idx = self.w2i['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[str(idx)]
        return {
            'input': torch.tensor(item['input']),
            'target': torch.tensor(item['target']),
            'length': item['length']
        }


# ======================================================
# Loss
# ======================================================
def loss_function(logp, target, length, mean, logv, pad_idx):
    max_len = torch.max(length).item()

    logp = logp[:, :max_len].contiguous().view(-1, logp.size(-1))
    target = target[:, :max_len].contiguous().view(-1)

    nll = torch.nn.NLLLoss(ignore_index=pad_idx, reduction='sum')(logp, target)
    kl = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

    return nll + kl, nll.item(), kl.item()

# ======================================================
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch
# ======================================================
# Train (Iteration-based)
# ======================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    preprocess_if_needed(args.data_dir, 'train',
                          args.vocab_file, args.max_sequence_length, args.min_occ)
    preprocess_if_needed(args.data_dir, 'valid',
                          args.vocab_file, args.max_sequence_length, args.min_occ)

    #File Name
    train_set = TextDataset(
        os.path.join(args.data_dir, 'rockyou-train.json'), 
        os.path.join(args.data_dir, args.vocab_file)
    )
    valid_set = TextDataset(
        os.path.join(args.data_dir, 'rockyou-valid.json'),
        os.path.join(args.data_dir, args.vocab_file)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    valid_loader = DataLoader(valid_set, batch_size=args.batch_size)

    model = PassDNVAE(
        vocab_size=len(train_set.w2i),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        sos_idx=train_set.w2i['<sos>'],
        eos_idx=train_set.w2i['<eos>'],
        pad_idx=train_set.pad_idx,
        max_sequence_length=args.max_sequence_length
    ).to(device)
    
        # ===== Print Parameters =====
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("====================================")
    print("Model architecture:\n", model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("====================================")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_gen = infinite_loader(train_loader)
    global_step = 0
    total_nll = total_kl = 0.0
    
    # ===== Best loss tracking =====
    best_total_loss = float('inf')
    best_nll = float('inf')
    best_kl = float('inf')
    best_step = 0
    best_time = 0

    model.train()

    while global_step < args.total_iters:
        batch = next(train_gen)

        x = batch['input'].to(device)
        y = batch['target'].to(device)
        length = torch.as_tensor(batch['length']).to(device)

        logp, mean, logv, _ = model(x, length)
        loss, nll, kl = loss_function(logp, y, length, mean, logv, model.pad_idx)
    

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_nll += nll
        total_kl += kl
        global_step += 1

        # ===== Logging =====
        if global_step % args.log_every == 0:
            print(
                f"[Iter {global_step}/{args.total_iters}] "
                f"NLL={total_nll/(args.log_every*args.batch_size):.4f} "
                f"KL={total_kl/(args.log_every*args.batch_size):.4f} "
                f"time={(time.time()-start_time):.2f}s"
            )
            total_nll = total_kl = 0.0

        # ===== Validation & Save =====
        if global_step % args.save_every == 0:
            model.eval()
            val_nll = val_kl = 0.0
            
            os.makedirs(args.save_dir, exist_ok=True)

            with torch.no_grad():
                for batch in valid_loader:
                    x = batch['input'].to(device)
                    y = batch['target'].to(device)
                    length = torch.as_tensor(batch['length']).to(device)

                    logp, mean, logv, _ = model(x, length)
                    _, nll, kl = loss_function(
                        logp, y, length, mean, logv, model.pad_idx
                    )

                    val_nll += nll
                    val_kl += kl

            val_nll /= len(valid_set)
            val_kl /= len(valid_set)

            print(f"[Validation @ {global_step}] NLL={val_nll:.4f} KL={val_kl:.4f}")
            
            # ===== Validation best =====
            val_total_loss = val_nll + val_kl

            if val_total_loss < best_total_loss:
                best_total_loss = val_total_loss
                best_nll = val_nll
                best_kl = val_kl
                best_step = global_step
                best_time = time.time() - start_time

                print(f"[BEST (VAL) UPDATE @ {global_step}] "
                    f"TOTAL={best_total_loss:.4f} "
                    f"NLL={best_nll:.4f} KL={best_kl:.4f}")
 
                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, "best_model.pt")
                )
            
            torch.save(
                model.state_dict(),
                os.path.join(args.save_dir, f"passdnvae_iter{global_step}.pt")
            )

            model.train()

    print(f"Training finished in {(time.time()-start_time):.2f}s")
    print("====================================")
    print("[BEST RESULT SUMMARY]")
    print(f"Best step: {best_step}")
    print(f"Best total loss: {best_total_loss:.4f}")
    print(f"Best NLL: {best_nll:.4f}")
    print(f"Best KL: {best_kl:.4f}")
    print(f"Time to best: {best_time:.2f}s")
    print(f"Optimization ratio (best/last): {best_total_loss / (nll + kl):.4f}")
    print("====================================")


# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data')  
    parser.add_argument('--vocab_file', default='rockyou-vocab.json') 
    parser.add_argument('--embedding_size', type=int, default=300)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--latent_size', type=int, default=64)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--embedding_dropout', type=float, default=0.1)
    parser.add_argument('--max_sequence_length', type=int, default=12)
    parser.add_argument('--min_occ', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # iteration-based
    parser.add_argument('--total_iters', type=int, default=200000)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)

    parser.add_argument('--save_dir', default='./rockyou')
    args = parser.parse_args()

    main(args)
