import torch, torch.nn as nn

class RM(nn.Module):
    def __init__(self, dim_text=384, dim_feat=4, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_text*2 + dim_feat, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
    def forward(self, emb_code, emb_review, feats):
        x = torch.cat([emb_code, emb_review, feats], dim=-1)
        return self.fc(x).squeeze(-1)
