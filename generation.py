import os
import argparse
import json
import torch
import time

from model import PassDNVAE

def idx2str(sample, i2w, pad, sos, eos):
    out = []

    for t in sample:
        t = int(t)

        if t == eos:
            break

        if t in (pad, sos):
            continue

        out.append(i2w.get(str(t), ""))

    return "".join(out)


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_path = os.path.join(
        args.data_dir,
        args.vocab_file,
    )

    output_path = os.path.abspath(args.output_file)
    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ==================================================
    # Vocabulary
    # ==================================================
    with open(
        vocab_path,
        "r",
        encoding="utf-8",
    ) as f:
        vocab = json.load(f)

    w2i = vocab["w2i"]
    i2w = vocab["i2w"]

    pad = w2i["<pad>"]
    sos = w2i["<sos>"]
    eos = w2i["<eos>"]

    # ==================================================
    # Model
    # ==================================================
    model = PassDNVAE(
        vocab_size=len(w2i),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        sos_idx=sos,
        eos_idx=eos,
        pad_idx=pad,
        max_sequence_length=args.max_sequence_length,
    )

    state = torch.load(
        args.model_path,
        map_location=device,
    )

    model.load_state_dict(state)

    model = model.to(device)
    model.eval()

    print("=" * 70)
    print(f"Device             : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU name           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(f"Model path         : {args.model_path}")
    print(f"Vocab path         : {vocab_path}")
    print(f"Target samples     : {args.num_samples:,}")
    print(f"Batch size         : {args.batch_size:,}")
    print(f"Unique mode        : {'ON' if args.unique else 'OFF'}")
    print("=" * 70)

    # ==================================================
    # Generation
    # ==================================================
    generated = 0
    sampled = 0
    duplicate_count = 0
    batch_count = 0

    gen_time = 0.0
    save_time = 0.0

    # Only allocate the set when --unique is enabled
    seen = set() if args.unique else None

    total_start = time.perf_counter()

    with open(
        output_path,
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as f:
        with torch.inference_mode():
            while generated < args.num_samples:

                # In normal mode, generate only the remaining number.
                # In unique mode, keep generating full batches because
                # some samples may be discarded as duplicates.
                if args.unique:
                    current_batch_size = args.batch_size
                else:
                    current_batch_size = min(
                        args.batch_size,
                        args.num_samples - generated,
                    )

                # --------------------------------------
                # GPU generation time
                # --------------------------------------
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                gen_start = time.perf_counter()

                samples, _ = model.inference(
                    n=current_batch_size
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                gen_time += (
                    time.perf_counter() - gen_start
                )

                sampled += current_batch_size

                # GPU tensor -> CPU
                samples = samples.detach().cpu().tolist()

                passwords = [
                    idx2str(
                        sample,
                        i2w,
                        pad,
                        sos,
                        eos,
                    )
                    for sample in samples
                ]

                # ==================================================
                # Unique filtering
                # ==================================================
                if args.unique:
                    unique_passwords = []

                    for password in passwords:
                        if password in seen:
                            duplicate_count += 1
                            continue

                        seen.add(password)
                        unique_passwords.append(password)

                    passwords = unique_passwords

                # Do not exceed the requested number of outputs
                remaining = args.num_samples - generated

                if len(passwords) > remaining:
                    passwords = passwords[:remaining]

                lines = [
                    password + "\n"
                    for password in passwords
                ]

                # --------------------------------------
                # File write time
                # --------------------------------------
                save_start = time.perf_counter()

                f.writelines(lines)

                save_time += (
                    time.perf_counter() - save_start
                )

                generated += len(lines)
                batch_count += 1

                if (
                    batch_count % args.log_every == 0
                    or generated >= args.num_samples
                ):
                    elapsed = (
                        time.perf_counter() - total_start
                    )

                    current_speed = (
                        generated / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    if args.unique:
                        duplicate_rate = (
                            duplicate_count / sampled * 100.0
                            if sampled > 0
                            else 0.0
                        )

                        print(
                            f"[{generated:,}/{args.num_samples:,}] "
                            f"Batch={batch_count:,} "
                            f"Sampled={sampled:,} "
                            f"Duplicates={duplicate_count:,} "
                            f"DupRate={duplicate_rate:.2f}% "
                            f"Time={elapsed:.2f}s "
                            f"Speed={current_speed:,.2f} unique/s"
                        )
                    else:
                        print(
                            f"[{generated:,}/{args.num_samples:,}] "
                            f"Batch={batch_count:,} "
                            f"Time={elapsed:.2f}s "
                            f"Speed={current_speed:,.2f} passwords/s"
                        )

    total_time = time.perf_counter() - total_start

    speed = (
        generated / total_time
        if total_time > 0
        else 0.0
    )

    # ==================================================
    # Result
    # ==================================================
    print("=" * 70)
    print(f"Output file        : {output_path}")
    print(f"Total generated    : {generated:,}")
    print(f"Target samples     : {args.num_samples:,}")
    print(f"Batch size         : {args.batch_size:,}")
    print(f"Total batch count  : {batch_count:,}")
    print(f"Total sampled      : {sampled:,}")

    if args.unique:
        duplicate_rate = (
            duplicate_count / sampled * 100.0
            if sampled > 0
            else 0.0
        )

        print(f"Duplicates removed : {duplicate_count:,}")
        print(f"Duplicate rate     : {duplicate_rate:.4f}%")

    print(f"Generation time    : {gen_time:.2f} seconds")
    print(f"File write time    : {save_time:.2f} seconds")
    print(f"Total time         : {total_time:.2f} seconds")

    if args.unique:
        print(
            f"Generation speed   : "
            f"{speed:.2f} unique passwords/second"
        )
    else:
        print(
            f"Generation speed   : "
            f"{speed:.2f} passwords/second"
        )

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Dataset directory path
    parser.add_argument(
        "--data_dir",
        default="/path/directory",
    )
    
    #Vocabulary file path
    parser.add_argument(
        "--vocab_file",
        default="/path/directory",
    )

    #Trained model file path
    parser.add_argument(
        "--model_path",
        default=(
            "/path/directory/passdnvae_best.pt"
        ),
    )

    #Generated password output file path
    parser.add_argument(
        "--output_file",
        default=(
            "/path/directory/PassDNVAE_rockyou.txt"
        ),
    )

    # Total number of outputs to generate
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1_000_000_000,
    )

    # Number of samples generated per inference call
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--latent_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--word_dropout",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--embedding_dropout",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
    )

    # Default:
    #   duplicates are allowed
    #
    # With --unique:
    #   duplicate outputs are removed
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Generate only unique outputs",
    )

    args = parser.parse_args()
    main(args)
