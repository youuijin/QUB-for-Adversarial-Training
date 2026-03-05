import torch
import torch.nn.functional as F

class FGSM_Attack():
    def __init__(self, model, eps=8.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def perturb(self, x_natural, y):
        self.model.eval()

        delta = torch.zeros_like(x_natural).to(self.device)

        delta.requires_grad_()

        output = self.model(x_natural + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.eps*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        self.model.train()

        return adv_x
    