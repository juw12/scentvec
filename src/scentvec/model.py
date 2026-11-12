"""The perfume encoder: a set of notes in, one unit-length vector out."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from .features import PAD, UNK


@dataclass
class ModelConfig:
    n_note_ids: int
    n_token_ids: int
    chem_dim: int
    n_accords: int
    n_families: int
    n_subfamilies: int
    hidden_dim: int = 128
    embed_dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    note_unk_dropout: float = 0.1
    use_note_ids: bool = True
    use_tokens: bool = True
    use_chem: bool = True
    use_positions: bool = True

    def to_json(self) -> dict:
        return asdict(self)


class PrototypeHead(nn.Module):
    """Cosine classifier whose class vectors live in the embedding space.

    Because the prototypes share the space with perfumes, "woody" or
    "Chypre" can later be used directly as a search query.
    """

    def __init__(self, n_classes: int, embed_dim: int, initial_scale: float = 10.0) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.1)
        self.log_scale = nn.Parameter(torch.tensor(float(initial_scale)).log())
        self.bias = nn.Parameter(torch.zeros(n_classes))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        return self.log_scale.exp() * embedding @ prototypes.T + self.bias

    def unit_prototypes(self) -> torch.Tensor:
        return F.normalize(self.prototypes.detach(), dim=-1)


class ScentEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.note_embedding = nn.Embedding(config.n_note_ids, hidden, padding_idx=PAD)
        self.token_embedding = nn.Embedding(config.n_token_ids, hidden, padding_idx=PAD)
        self.chem_projection = nn.Sequential(
            nn.Linear(config.chem_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.position_embedding = nn.Embedding(4, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.input_dropout = nn.Dropout(config.dropout)

        if config.n_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=config.n_heads,
                dim_feedforward=2 * hidden,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.context = nn.TransformerEncoder(
                layer, num_layers=config.n_layers, enable_nested_tensor=False
            )
        else:
            # DeepSets variant: each note is transformed on its own.
            self.context = None
            self.note_mlp = nn.Sequential(
                nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
            )
        self.output_norm = nn.LayerNorm(hidden)
        self.attention_score = nn.Linear(hidden, 1)
        self.projection = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, config.embed_dim)
        )

        self.accord_head = PrototypeHead(config.n_accords, config.embed_dim)
        self.family_head = PrototypeHead(config.n_families, config.embed_dim)
        self.subfamily_head = PrototypeHead(config.n_subfamilies, config.embed_dim)

    def note_vectors(
        self,
        note_ids: torch.Tensor,
        token_ids: torch.Tensor,
        chem: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        vectors = torch.zeros(
            note_ids.shape + (config.hidden_dim,), dtype=torch.float32, device=note_ids.device
        )
        if config.use_note_ids:
            if self.training and config.note_unk_dropout > 0:
                # Hide some note identities so the word and chemistry channels
                # learn to stand in for notes the model has never seen.
                hide = torch.rand(note_ids.shape, device=note_ids.device) < config.note_unk_dropout
                note_ids = torch.where(hide & (note_ids != PAD), torch.full_like(note_ids, UNK), note_ids)
            vectors = vectors + self.note_embedding(note_ids)
        if config.use_tokens:
            token_vectors = self.token_embedding(token_ids)
            token_count = (token_ids != PAD).sum(dim=-1, keepdim=True).clamp(min=1)
            vectors = vectors + token_vectors.sum(dim=-2) / token_count
        if config.use_chem:
            vectors = vectors + self.chem_projection(chem)
        if config.use_positions:
            vectors = vectors + self.position_embedding(positions)
        return self.input_dropout(self.input_norm(vectors))

    def forward(
        self,
        note_ids: torch.Tensor,
        token_ids: torch.Tensor,
        chem: torch.Tensor,
        positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        vectors = self.note_vectors(note_ids, token_ids, chem, positions)
        if self.context is not None:
            hidden = self.context(vectors, src_key_padding_mask=~mask)
        else:
            hidden = vectors + self.note_mlp(vectors)
        hidden = self.output_norm(hidden)

        mask_float = mask.unsqueeze(-1).float()
        mean_pooled = (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        scores = self.attention_score(hidden).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention_pooled = (hidden * weights).sum(dim=1)

        embedding = self.projection(torch.cat([mean_pooled, attention_pooled], dim=-1))
        return F.normalize(embedding, dim=-1)

    def heads(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "accords": self.accord_head(embedding),
            "family": self.family_head(embedding),
            "subfamily": self.subfamily_head(embedding),
        }


def to_tensors(batch: Dict[str, "object"], device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {name: torch.from_numpy(array).to(device) for name, array in batch.items()}
