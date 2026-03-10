from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EmbeddingKNNModel:
    embeddings: torch.Tensor
    labels: torch.Tensor
    k: int = 8
    metric: str = "euclidean"

    def predict(self, query_embeddings: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            emb = F.normalize(self.embeddings, dim=-1)
            qry = F.normalize(query_embeddings, dim=-1)
            distances = 1.0 - qry @ emb.T
        else:
            distances = torch.cdist(query_embeddings, self.embeddings, p=2)

        k = min(self.k, self.embeddings.size(0))
        idx = torch.topk(distances, k=k, largest=False, dim=1).indices
        neigh_labels = self.labels[idx]

        preds = []
        for row in neigh_labels:
            bincount = torch.bincount(row, minlength=10)
            preds.append(int(torch.argmax(bincount).item()))
        return torch.tensor(preds, device=query_embeddings.device)
