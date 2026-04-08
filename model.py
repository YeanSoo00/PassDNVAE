import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------
# DenseBlock
# --------------------------
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=9, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(in_channels + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(
                        in_channels + i * growth_rate,
                        growth_rate,
                        kernel_size=3,
                        padding=1,
                    ),
                )
            )
        self.out_channels = in_channels + growth_rate * n_layers

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))
            features.append(out)
        return torch.cat(features, dim=1)


# ======================================================
# PassDNVAE (temperature 제거 버전)
# ======================================================
class PassDNVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        word_dropout,
        embedding_dropout,
        latent_size,
        sos_idx,
        eos_idx,
        pad_idx,
        max_sequence_length,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.word_dropout_rate = word_dropout
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.max_sequence_length = max_sequence_length

        # ---------------- Encoder ----------------
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder_conv = nn.Conv1d(
            embedding_size, hidden_size, kernel_size=5, padding=2
        )
        self.denseblock = DenseBlock(hidden_size, growth_rate=9, n_layers=3)

        self.attn = nn.Linear(self.denseblock.out_channels, 1)
        self.pool_logits = nn.Parameter(torch.zeros(3))

        self.hidden2mean = nn.Linear(self.denseblock.out_channels, latent_size)
        self.hidden2logv = nn.Linear(self.denseblock.out_channels, latent_size)

        # ---------------- Decoder ----------------
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.decoder_rnn = nn.GRU(
            input_size=embedding_size + latent_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.outputs2vocab = nn.Linear(hidden_size, vocab_size)

    # ---------------- ENCODER ----------------
    def encode(self, input_sequence, length):
        B, L = input_sequence.size()
        device = input_sequence.device

        emb = self.embedding_dropout(self.embedding(input_sequence))
        x = self.encoder_conv(emb.transpose(1, 2))
        x = self.denseblock(x).transpose(1, 2)

        mask = torch.arange(L, device=device).unsqueeze(0) < length.unsqueeze(1)

        attn_score = self.attn(x).squeeze(-1)
        attn_score = attn_score.masked_fill(~mask, -1e9)
        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(-1)
        f_att = torch.sum(attn_weight * x, dim=1)

        x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
        f_max = torch.max(x_masked, dim=1).values
        f_avg = (x * mask.unsqueeze(-1)).sum(dim=1) / length.unsqueeze(1)

        alpha, beta, gamma = torch.softmax(self.pool_logits, dim=0)
        f_final = alpha * f_att + beta * f_max + gamma * f_avg

        mean = self.hidden2mean(f_final)
        logv = self.hidden2logv(f_final)
        std = torch.exp(0.5 * logv)
        z = mean + std * torch.randn_like(std)

        return z, mean, logv

    # ---------------- FORWARD ----------------
    def forward(self, input_sequence, length):
        z, mean, logv = self.encode(input_sequence, length)

        hidden = self.latent2hidden(z).unsqueeze(0).repeat(2, 1, 1)

        if self.word_dropout_rate > 0:
            prob = torch.rand_like(input_sequence.float())
            mask = (
                (prob < self.word_dropout_rate)
                & (input_sequence != self.pad_idx)
                & (input_sequence != self.sos_idx)
            )
            input_sequence = input_sequence.masked_fill(mask, self.pad_idx)

        emb = self.embedding_dropout(self.embedding(input_sequence))

        z_expand = z.unsqueeze(1).expand(-1, emb.size(1), -1)
        decoder_input = torch.cat([emb, z_expand], dim=-1)

        out, _ = self.decoder_rnn(decoder_input, hidden)
        logp = F.log_softmax(self.outputs2vocab(out), dim=-1)

        return logp, mean, logv, z

    # ---------------- Sampling (temperature 제거) ----------------
    def _sample(self, logits, k=30):
        values, indices = torch.topk(logits, k, dim=-1)
        probs = F.softmax(values, dim=-1)
        idx = torch.multinomial(probs, 1)
        return indices.gather(-1, idx).squeeze(-1)

    # ---------------- Inference ----------------
    @torch.no_grad()
    def inference(self, n=1, k=30):
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_size, device=device)
        hidden = self.latent2hidden(z).unsqueeze(0).repeat(2, 1, 1)

        seq = torch.full((n, 1), self.sos_idx, device=device, dtype=torch.long)
        output = torch.full(
            (n, self.max_sequence_length), self.pad_idx, device=device
        )

        active = torch.ones(n, dtype=torch.bool, device=device)

        for t in range(self.max_sequence_length):
            emb = self.embedding(seq)
            z_expand = z.unsqueeze(1)
            decoder_input = torch.cat([emb, z_expand], dim=-1)

            out, hidden = self.decoder_rnn(decoder_input, hidden)
            logits = self.outputs2vocab(out[:, -1])

            next_token = self._sample(logits, k)
            next_token[~active] = self.pad_idx
            output[:, t] = next_token

            active &= next_token != self.eos_idx
            if not active.any():
                break

            seq = next_token.unsqueeze(1)

        return output, z
