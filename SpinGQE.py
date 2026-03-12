import torch
from torch.nn import functional as F
from model import GPT


class SpinGQE(GPT):
    def forward(self, idx):
        device = idx.device
        batch_size, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # forward the GPT model itself
        token_emb = self.transformer.wte(idx)  # token embeddings of shape (batch_size, seq_len, n_embd)
        position_emb = self.transformer.wpe(pos)  # position embedding of shape (seq_len, n_embd)
        x = self.transformer.drop(token_emb + position_emb)  # dropout/regularization
        for block in self.transformer.h:  # applies each transformer block
            x = block(x)
        x = self.transformer.ln_f(x)  # final layer normalization layer
        logits = self.lm_head(x)  # convert layer to logits
        return logits

    def calculate_loss(self, tokens, energies, b1=None):
        current_tokens, next_tokens = tokens[:, :-1], tokens[:, 1:]
        # calculate the logits for the next possible tokens in the sequence
        logits = self(current_tokens)
        # get the logit for the actual next token in the sequence
        next_token_mask = torch.nn.functional.one_hot(
            next_tokens, num_classes=self.config.vocab_size
        )
        next_token_logits = (logits * next_token_mask).sum(axis=2)
        # calculate the cumulative logits for each subsequence
        cumsum_logits = torch.cumsum(next_token_logits, dim=1)
        # match cumulative logits to subsequence energies

        if b1 is None:
            loss = torch.mean(torch.square(cumsum_logits - energies))
        else:
            weights = 1 / (1 + torch.exp(b1 * energies))  # freeze gradient for true energy
            # weights = weights / weights.sum() * len(weights)
            # loss = torch.mean(weights * torch.exp(b2 * torch.abs(cumsum_logits - energies)))
            loss = torch.mean(weights * torch.square(cumsum_logits - energies))
            uw_loss = torch.mean(torch.square(cumsum_logits - energies))

        return loss, uw_loss

    @torch.no_grad()
    def generate(self, n_sequences, max_new_tokens, temperature=1.0, device="cpu"):
        idx = torch.zeros(size=(n_sequences, 1), dtype=int, device=device)
        total_logits = torch.zeros(size=(n_sequences, 1), device=device)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]  # crops sequence to blocksize
            logits = self(idx_cond)  # forward the model to get logits forthe index in the sequence
            logits = logits[:, -1, :]  # pluck logits from final step
            logits[:, 0] = float('inf')  # sets logit of first token so probability 0sampled_ids)
            probs = F.softmax(-logits / temperature, dim=-1)  # apply softmax to get probabilities from logits 
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from probability distribution
            total_logits += torch.gather(logits, index=idx_next, dim=1)  # accumulates logits
            idx = torch.cat((idx, idx_next), dim=1)  # append sampled index to sequence
        return idx, total_logits