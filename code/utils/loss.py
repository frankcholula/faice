import torch
import torch.nn.functional as F
import lpips
def get_loss(pred, target, config, lpips_fn=None):
    if config.loss_type == "mse":
        loss = F.mse_loss(pred, target)
    elif config.loss_type == "l1":
        loss = F.l1_loss(pred, target)
    else:
        raise ValueError("Unsupported base loss type")

    if config.use_lpips_regularization:
        if lpips_fn is None:
            lpips_fn = lpips.LPIPS(net=config.lpips_net).to(pred.device)
            lpips_fn.eval()
        with torch.no_grad():
            lpips_loss = lpips_fn(pred, target).mean()
        loss = loss + config.lpips_weight * lpips_loss

    return loss

