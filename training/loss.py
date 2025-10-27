import torch

def mutual_information(X: torch.Tensor, templates: torch.Tensor, n, device='cpu'):
    atol = 1e-2
    rtol = 1e-2

    # Obtain max activation
    X_flat = X.view(X.shape[0], X.shape[1], -1)
    indices = torch.argmax(X_flat, dim=-1)

    X_expanded = X.unsqueeze(2)
    T_expanded = templates.unsqueeze(0).unsqueeze(0)
    T_expanded = T_expanded[:,:, indices, :, :].view(X_expanded.shape)

    XT = torch.mul(X_expanded, T_expanded).squeeze(2)

    # XT = XT / s_max.unsqueeze(-1).unsqueeze(-1)
    trace = XT.sum(dim=(-1, -2))
    exp = torch.exp(trace)

    ZT = exp.sum(dim=-1).unsqueeze(-1)

    pXT = exp / ZT

    one = torch.tensor(1.0, device=pXT.device, dtype=pXT.dtype)
    # assert torch.allclose(pXT.sum(dim=-1), one, rtol=rtol, atol=atol), "sum_x p(x|T) debe ser 1"
    
    # pT = T_expanded.sum(dim=-1).sum(dim=-1).unsqueeze(-1)
    # pX = ((torch.mul(pXT,pT)).sum(dim=1)) 
    # sum_pX = pX.nansum(dim=1)
    # one = torch.tensor(1.0, device=pT.device, dtype=pT.dtype)
    # try:
    #     assert torch.allclose(sum_pX, one, rtol=rtol, atol=atol), f"sum_x p(x) debe ser 1, pero es {sum_pX}"
    # except AssertionError as e:
    #     print(e)

    # eps_box = 1e-7
    # try:
    #     assert (pXT >= -eps_box).all() and (pXT <= 1+eps_box).all(), f"Valores fuera de [0,1] en p(x|T)"
    #     assert (pX  >= -eps_box).all() and (pX  <= 1+eps_box).all(), f"Valores fuera de [0,1] en p(x)"
    # except AssertionError as e:
    #     print(e)
    # eps = 1e-12
    # ratio = pXT.clamp_min(eps) / pX.unsqueeze(1).clamp_min(eps)
    MI = - ( (pXT).nansum(dim=-2)).nansum(dim=-1)
    return MI