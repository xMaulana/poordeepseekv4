import torch
import torch.nn.functional as F

x = torch.zeros(1, 4, device='cuda', requires_grad=True)
scores = x.masked_fill(torch.ones_like(x, dtype=torch.bool), float("-inf"))
probs = F.softmax(scores, dim=-1)
probs = torch.nan_to_num(probs)
out = probs.sum()
out.backward()
print("Grad:", x.grad)
