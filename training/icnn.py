import torch
from torch import nn
import logging

logging.basicConfig(level=logging.INFO)

def generate_templates(n: int, tau: float = 0.5, beta: float = 4.0):
    """
    Generates n^2 positive templates and one negative template
    for an interpretable CNN, as described in Zhang et al. (2018).

    Each positive template corresponds to a potential spatial position
    in an n x n feature map. Values decay according to the L1 distance
    from the central position. A negative template is also generated
    to represent the absence of activation.

    Args:
        n (int): Size of the feature map (n x n).
        tau (float, optional): Scaling constant for maximum activation value.
            Defaults to 0.5.
        beta (float, optional): Decay factor based on L1 distance.
            Defaults to 4.0.

    Returns:
        tuple:
            templates_pos (list[torch.Tensor]): List of n^2 positive templates
                (each of shape n x n).
            template_neg (torch.Tensor): Negative template of shape n x n
                filled with -tau.

    Example:
        >>> templates_pos, template_neg = generate_templates(4)
        >>> len(templates_pos)
        16
        >>> template_neg.shape
        torch.Size([4, 4])
    """
    templates_pos = []
    coords = [(i, j) for i in range(n) for j in range(n)]

    for mu in coords:
        template = torch.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist = abs(i - mu[0]) + abs(j - mu[1])  # L1 distance
                value = tau * max(1 - beta * dist / n, -1)
                template[i, j] = value
        templates_pos.append(template)

    template_neg = torch.full((n, n), -tau)

    templates = torch.stack(templates_pos + [template_neg], dim=0)
    return templates


class InterpretableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, feature_map_size, tau=0.5, beta=4.0, alpha=0.95):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=True)
        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.n = feature_map_size
        self.templates = generate_templates(self.n, tau=self.tau, beta=self.beta)
        # Templates are not trainable
        self.templates.requires_grad = False
        self.negative_template = self.templates[-1]

    def forward(self, x):
        x = nn.ReLU()(self.conv(x))
        if not self.training:
            # logging.info("Inference mode: applying templates to feature maps.")
            # negative_template_stack = torch.repeat(self.negative_template.unsqueeze(0), x.shape[0]*x.shape[1], 1, 1)
            masks = self._assign_masks(x)
            masked_x = x * masks
            return masked_x * (masked_x > 0)
        else:
            logging.info("Training mode: returning raw feature maps.")
            return x

    def _assign_masks(self, x):

        B, C, H, W = x.shape

        x_flat = x.view(B * C, H, W)
        masks = []

        for fmap in x_flat:
            mu_hat = torch.argmax(fmap.view(-1))
            i, j = mu_hat // self.n, mu_hat % self.n
            idx = i * self.n + j
            mask = self.templates[idx]
            masks.append(mask)

        masks = torch.stack(masks).view(B, C, H, W).to(x.device)
        return masks