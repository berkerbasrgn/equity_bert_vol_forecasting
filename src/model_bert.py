# model_equitybert.py
#
# Adapted from model_bert.py (Vola-BERT) for S&P 500 equity index forecasting.
#
# Renaming rationale:
#   The original class is named Vola_BERT and is designed for FX markets.
#   This adaptation targets equity index volatility, hence the name EquityBERT.
#   The architecture is identical to Vola_BERT; the rename makes it clear in
#   experiment logs and thesis text which model is being run.
#
# What is unchanged from the original:
#   - BERT backbone: bert-base-uncased loaded from HuggingFace
#   - Freezing strategy: attention.self.{query,key,value},
#     attention.output.dense, intermediate.dense, output.dense are frozen.
#     LayerNorm, positional embeddings, wte, semantic embeddings, and head
#     are all trainable.
#   - Input encoding: one linear projection per feature (one-feature-one-token)
#   - Token selection: semantic token outputs + last feature token concatenated
#   - RevIN normalisation
#   - Forecast head: linear layer over concatenated selected tokens
#   - Weight initialisation scheme (mean=0, std=0.02 from BERT pre-training)
#
# What changed:
#   - Class renamed from Vola_BERT to EquityBERT
#   - semantic_tokens dict now expects keys matching the SP500 token vocabulary:
#       {"market_session": 4, "event_type": 5, "event_impact": 3}
#     instead of the original {"event": 3, "session": 5}
#   - Docstrings updated to reflect equity context and the r_t target variable

import torch
import torch.nn as nn
from transformers import BertModel


class EquityBERT(nn.Module):
    """
    BERT-based model fine-tuned for equity index volatility forecasting.

    Adapted from Vola_BERT (Nguyen et al., ICAIF 2025) for S&P 500 hourly data.
    The architecture follows the same three-stage pipeline:

        Stage 1 — Input Encoding:
            Each of the N numerical feature time series (length seq_len) is
            projected to a 768-dim embedding vector via a shared linear layer.
            This gives one token per feature (one-feature-one-token design).

        Stage 2 — BERT Encoder with PEFT:
            Semantic tokens (market session, event type, event impact) are
            prepended to the sequence of feature tokens. The combined sequence
            is processed by the first n_layer layers of bert-base-uncased.
            Core attention and feedforward weights are frozen; only LayerNorm,
            positional embeddings, the input projection, the semantic token
            embeddings, and the forecast head are trained.

        Stage 3 — Linear Probing:
            The hidden states of the semantic tokens and the last feature token
            are concatenated and passed through a linear layer to produce the
            pred_len-step forecast of r_t = ln(H_t / L_t).

    Notation:
        B  : batch size
        N  : number of numerical feature time series (num_series)
        S  : number of semantic token types
        E  : BERT hidden dimension (768)
        L  : lookback horizon (seq_len)
        P  : forecast horizon (pred_len)
    """

    def __init__(
        self,
        num_series: int,
        input_len: int,
        pred_len: int,
        n_layer: int,
        revin: bool = True,
        head_drop_rate: float = 0.2,
        semantic_tokens: dict = None,
    ):
        """
        Arguments:
            num_series (int)         : number of numerical input features N
            input_len (int)          : lookback window length L (e.g. 48)
            pred_len (int)           : forecast horizon P (e.g. 12)
            n_layer (int)            : number of BERT encoder layers to use
                                       (hyperparameter; sweep over {2, 4, 6})
            revin (bool)             : use Reversible Instance Normalisation
            head_drop_rate (float)   : dropout rate before the forecast head
            semantic_tokens (dict)   : maps token name -> vocabulary size, e.g.
                                       {"market_session": 4,
                                        "event_type": 5,
                                        "event_impact": 3}
                                       Must match TOKEN_MAPPINGS in dataset_sp500.py
        """
        super().__init__()

        if semantic_tokens is None:
            raise ValueError(
                "semantic_tokens must be provided. "
                "Pass SEMANTIC_TOKEN_VOCAB from dataset_sp500.py, "
                "or an empty dict {} to disable semantic conditioning."
            )

        self.revin     = revin
        self.num_series = num_series
        self.input_len = input_len
        self.pred_len  = pred_len
        self.n_layer   = n_layer

        # BERT backbone
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Use only the first n_layer transformer blocks
        self.bert.encoder.layer = self.bert.encoder.layer[:n_layer]
        self.n_embd = self.bert.config.hidden_size  # 768

        # Stage 1: Input encoder
        # One linear layer projects the full time series of each feature
        # (length input_len) into a single 768-dim embedding vector.
        # Unchanged from original.
        self.wte = nn.Linear(self.input_len, self.n_embd)
        self.wte.apply(self._init_weights)

        # Stage 2: PEFT freezing strategy
        # Freeze core attention and feedforward weights.
        # Trainable: LayerNorm, positional embeddings, wte, semantic
        #            token embeddings, forecast head.
        # Unchanged from original.
        _freeze_keywords = [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense",
            "intermediate.dense",
            "output.dense",
        ]
        for name, param in self.bert.named_parameters():
            if any(kw in name for kw in _freeze_keywords):
                param.requires_grad = False

        # Semantic token embeddings
        # Change from original: keys are now 'market_session', 'event_type',
        # 'event_impact' (matching the SP500 TOKEN_MAPPINGS vocabulary)
        # instead of the original 'session' and 'event' keys.
        _sem_embeds = {}
        self.token_orders = list(semantic_tokens.keys())  # Store token order for later selection in forward()
        for token_name, vocab_size in semantic_tokens.items():
            emb = nn.Embedding(vocab_size, self.n_embd)
            emb.apply(self._init_weights)
            _sem_embeds[token_name] = emb
        self.semantic_token_embeddings = nn.ModuleDict(_sem_embeds)

        # Stage 3: Forecast head
        # Input: (S + 1) * E  where S = number of semantic token types
        #        and +1 is the last feature token
        # Output: pred_len scalar forecasts
        # Unchanged from original.
        n_sem = len(semantic_tokens)
        self.head = nn.Linear(
            self.n_embd * (n_sem + 1), self.pred_len, bias=True
        )
        self.head.apply(self._init_weights)
        self.head_drop = nn.Dropout(head_drop_rate)

    def _init_weights(self, module):
        """
        Initialises weights for components trained from scratch.
        Mean=0, std=0.02 follows the BERT pre-training convention.
        Unchanged from original.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    def encoder(self, x: torch.Tensor, input_tokens: dict) -> torch.Tensor:
        """
        Runs the BERT encoder over the combined sequence of feature and
        semantic token embeddings.

        Arguments:
            x            : (B, N, L) numerical features
            input_tokens : dict mapping token name -> (B,) integer tensor

        Returns:
            h : (B, N+S, E) contextualised hidden states
        """
        # Build list of embeddings: semantic tokens first, then features
        embedding_list = []
        for token_name in self.token_orders:
            tok_vals = input_tokens[token_name]                          # (B,)
            emb = self.semantic_token_embeddings[token_name](tok_vals)   # (B, E)
            embedding_list.append(emb.unsqueeze(1))                      # (B, 1, E)

        # Feature embeddings: project each series' time series to E dims
        embedding_list.append(self.wte(x))   # (B, N, E)

        tok_emb = torch.cat(embedding_list, dim=1)   # (B, N+S, E)

        h = self.bert(
            inputs_embeds=tok_emb, attention_mask=None
        ).last_hidden_state                          # (B, N+S, E)

        return h

    def forward(self, x_data: tuple) -> torch.Tensor:
        """
        Forward pass.

        Arguments:
            x_data : tuple of (x, input_tokens)
                x            : (B, N, L) numerical features
                input_tokens : dict of token tensors

        Returns:
            Forecast tensor of shape (B, 1, pred_len)
        """
        x, input_tokens = x_data

        # ---- RevIN: instance normalisation per sample -------------------
        if self.revin:
            x = x.permute(0, 2, 1)             # (B, L, N)
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x = x / stdev
            x = x.permute(0, 2, 1)             # (B, N, L)

        # ---- Encoder (Stage 1 + 2) 
        h = self.encoder(x, input_tokens)       # (B, N+S, E)
        B = h.shape[0]

        # ---- Token selection (Stage 3 input)
        # Take the S semantic token positions (prepended at the front)
        # and the very last feature token (position -1).
        # Rationale: in bidirectional BERT, the last feature token aggregates
        # context from all feature tokens; semantic tokens encode regime info.
        n_sem = len(self.semantic_token_embeddings)
        token_indices = list(range(n_sem)) + [-1]
        h_selected = h[:, token_indices, :]                       # (B, S+1, E)
        h_flat = h_selected.reshape(B, (n_sem + 1) * self.n_embd) # (B, (S+1)*E)

        # ---- Forecast head ----------------------------------------------
        h_flat  = self.head_drop(h_flat)
        dec_out = self.head(h_flat)             # (B, pred_len)

        # ---- RevIN denormalisation --------------------------------------
        if self.revin:
            target_stdev = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, -1]
            target_mean  = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)[:, :, -1]
            dec_out = dec_out * target_stdev + target_mean

        return dec_out.unsqueeze(1)             # (B, 1, pred_len)

    @property
    def num_params(self):
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}