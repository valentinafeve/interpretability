import torch
from training.utils.plots import plot_tensors

def mutual_information(X: torch.Tensor, templates: torch.Tensor, n, n_classes, labels, device='cpu'):
    atol = 1e-2
    rtol = 1e-2

    # Obtain max activation
    X_flat = X.view(X.shape[0], X.shape[1], -1)
    indices = torch.argmax(X_flat, dim=-1)

    X_expanded = X.unsqueeze(2)
    T_expanded = templates.unsqueeze(0).unsqueeze(0)[:, :, :-1, :, :]
    T_negative = templates.unsqueeze(0).unsqueeze(0)[:, :, -1, :, :]
    T_expanded = T_expanded[:, :, indices, :, :].view(X_expanded.shape)

    XT = torch.mul(X_expanded, T_expanded)
    XT_negative = torch.mul(X_expanded, T_negative)
    XT_stack = torch.stack([XT, XT_negative])

    filters = XT.shape[1]
    batch = XT.shape[0]
    indexes = torch.arange(filters // n_classes, device=XT_stack.device, dtype=torch.long).repeat((batch, 1)) + (labels * (filters // n_classes)).unsqueeze(1) 
    indexes_one_hot = torch.nn.functional.one_hot(indexes, num_classes=filters).sum(dim=1)
    mask = indexes_one_hot[:, :, None, None, None].expand_as(XT).bool()
    mask_inverted = ~mask

    XT = torch.mul(XT, mask)
    XT_negative = torch.mul(XT_negative, mask_inverted)

    XT_combined = XT + XT_negative

    s_max = XT_combined.max(dim=-1)[0].max(dim=-1)[0]
    s_min = XT_combined.min(dim=-1)[0].min(dim=-1)[0]

    # XT = XT / s_max.unsqueeze(-1).unsqueeze(-1)
    trace = XT_combined.nansum(dim=(-1, -2))           # (B, C, T)

    # numerically stable softmax over dim=1 (can confirm below)
    trace_max, _ = trace.max(dim=1, keepdim=True)   # (B, 1, T)
    stable_trace = trace - trace_max                # shift so max is 0

    exp = torch.exp(stable_trace)                  # safe now
    ZT = exp.sum(dim=1, keepdim=True)              # (B, 1, T)
    pXT = exp / (ZT + 1e-12)                       # (B, C, T)

    pT = torch.full((1, (n*n+1)), 1/(1+n*n)).to(device) # (1 , 257)
    pX = ((torch.mul(pXT,pT)).sum(dim=1)) # (2)
    sum_pX = pX.nansum(dim=1)
    one = torch.tensor(1.0, device=pT.device, dtype=pT.dtype)
    try:
        assert torch.allclose(sum_pX, one, rtol=rtol, atol=atol), f"sum_x p(x) debe ser 1, pero es {sum_pX}"
    except AssertionError as e:
        print(e)

    eps_box = 1e-7
    try:
        assert (pXT >= -eps_box).all() and (pXT <= 1+eps_box).all(), f"Valores fuera de [0,1] en p(x|T)"
        assert (pX  >= -eps_box).all() and (pX  <= 1+eps_box).all(), f"Valores fuera de [0,1] en p(x)"
    except AssertionError as e:
        print(e)
    eps = 1e-12
    ratio = pXT.clamp_min(eps) / pX.unsqueeze(1).clamp_min(eps)
    MI = - ((pT) * (pXT * torch.log(ratio)).nansum(dim=-2)).nansum(dim=-1)
    return MI