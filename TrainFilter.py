import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from PSDDataset import PSDDataset
from FilterModel import FilterEstimator
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt


def setup_logging(output_dir):
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger('filter_training')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger


def compute_psd(image, device):
    """Compute normalized PSD from image tensor."""
    with torch.no_grad():
        if image.dim() == 4:
            image = image[:, 0, :, :]  # Remove channel dim
        fft = torch.fft.fft2(image)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        psd = torch.abs(fft_shifted) ** 2
        psd = torch.log(psd + 1)

        # Normalize per sample
        B = psd.shape[0]
        psd_flat = psd.view(B, -1)
        psd_min = psd_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        psd_max = psd_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        psd = (psd - psd_min) / (psd_max - psd_min + 1e-10)

        return psd.unsqueeze(1)  # [B, 1, H, W]


def compute_fft(image, device='cuda'):
    """Compute shifted FFT of image."""
    if image.dim() == 4:
        image = image[:, 0, :, :]
    fft = torch.fft.fft2(image)
    return torch.fft.fftshift(fft, dim=(-2, -1))


def compute_real_filter(I_smooth_fft, I_sharp_fft):
    """Compute the actual filter ratio from FFT of image pairs."""
    eps = 1e-10
    filter_s2sh = I_sharp_fft / (I_smooth_fft + eps)
    filter_sh2s = I_smooth_fft / (I_sharp_fft + eps)

    # Take magnitude (real filters)
    filter_s2sh = torch.abs(filter_s2sh)
    filter_sh2s = torch.abs(filter_sh2s)

    return filter_s2sh, filter_sh2s


def gaussian_blur_2d(x, kernel_size=15, sigma=3.0):
    """Apply Gaussian blur to smooth the filter."""
    if x.dim() == 3:
        x = x.unsqueeze(1)
    pad = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - pad
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.outer(g).unsqueeze(0).unsqueeze(0)
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.conv2d(x_padded, kernel).squeeze(1)


def generate_images(I_smooth, I_sharp, filter_s2sh, filter_sh2s, device):
    """Generate images using the predicted filters."""
    I_smooth_fft = compute_fft(I_smooth, device)
    I_sharp_fft = compute_fft(I_sharp, device)

    # Apply filters
    gen_sharp_fft = I_smooth_fft * filter_s2sh
    gen_smooth_fft = I_sharp_fft * filter_sh2s

    # Inverse FFT
    gen_sharp_fft = torch.fft.ifftshift(gen_sharp_fft, dim=(-2, -1))
    gen_smooth_fft = torch.fft.ifftshift(gen_smooth_fft, dim=(-2, -1))

    I_gen_sharp = torch.fft.ifft2(gen_sharp_fft).real
    I_gen_smooth = torch.fft.ifft2(gen_smooth_fft).real

    return torch.clamp(I_gen_sharp, 0, 1).unsqueeze(1), torch.clamp(I_gen_smooth, 0, 1).unsqueeze(1)


def plot_filters(pred_s2sh, pred_sh2s, real_s2sh, real_sh2s, epoch, output_dir):
    """Plot filter comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Take center row for 1D visualization
    center = pred_s2sh.shape[-1] // 2

    axes[0, 0].plot(pred_s2sh[0, center, :].cpu().numpy())
    axes[0, 0].set_title('Predicted S2SH')
    axes[0, 0].set_ylim([0, 5])

    axes[0, 1].plot(real_s2sh[0, center, :].cpu().numpy())
    axes[0, 1].set_title('Real S2SH')
    axes[0, 1].set_ylim([0, 5])

    axes[1, 0].plot(pred_sh2s[0, center, :].cpu().numpy())
    axes[1, 0].set_title('Predicted SH2S')
    axes[1, 0].set_ylim([0, 2])

    axes[1, 1].plot(real_sh2s[0, center, :].cpu().numpy())
    axes[1, 1].set_title('Real SH2S')
    axes[1, 1].set_ylim([0, 2])

    plt.suptitle(f'Filter Comparison - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(output_dir / f'filter_epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def plot_images(I_smooth, I_sharp, I_gen_sharp, I_gen_smooth, epoch, output_dir):
    """Plot image comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(I_smooth[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Input Smooth')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(I_sharp[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Input Sharp')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(I_gen_sharp[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Generated Sharp')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(I_gen_smooth[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Generated Smooth')
    axes[1, 1].axis('off')

    plt.suptitle(f'Image Reconstruction - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(output_dir / f'images_epoch_{epoch:03d}.png', dpi=100)
    plt.close()


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    total_ft_loss = 0
    total_recon_loss = 0
    n_batches = 0

    l1_loss = nn.L1Loss()

    for I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2 in tqdm(dataloader, desc=f"Epoch {epoch}"):
        I_smooth = I_smooth_1.to(device)
        I_sharp = I_sharp_1.to(device)

        # Compute PSDs
        psd_smooth = compute_psd(I_smooth, device)
        psd_sharp = compute_psd(I_sharp, device)

        # Compute real filter from FFT
        I_smooth_fft = compute_fft(I_smooth, device)
        I_sharp_fft = compute_fft(I_sharp, device)
        real_s2sh, real_sh2s = compute_real_filter(I_smooth_fft, I_sharp_fft)

        # Smooth the real filter targets
        real_s2sh_smooth = gaussian_blur_2d(real_s2sh)
        real_sh2s_smooth = gaussian_blur_2d(real_sh2s)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
            # Predict filters
            pred_s2sh, pred_sh2s = model(psd_smooth, psd_sharp)

            # Filter loss (compare to smoothed real filter)
            ft_loss = l1_loss(pred_s2sh, real_s2sh_smooth) + l1_loss(pred_sh2s, real_sh2s_smooth)

            # Generate images
            I_gen_sharp, I_gen_smooth = generate_images(
                I_smooth, I_sharp, pred_s2sh, pred_sh2s, device
            )

            # Reconstruction loss
            recon_loss = (l1_loss(I_gen_sharp, I_sharp) + l1_loss(I_gen_smooth, I_smooth)) / 2.0

            loss = ft_loss + recon_loss

        optimizer.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_ft_loss += ft_loss.item()
        total_recon_loss += recon_loss.item()
        n_batches += 1

    return {
        'total_loss': total_loss / n_batches,
        'ft_loss': total_ft_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_ft_loss = 0
    total_recon_loss = 0
    n_batches = 0

    l1_loss = nn.L1Loss()
    vis_data = None

    for I_smooth_1, I_sharp_1, I_smooth_2, I_sharp_2 in dataloader:
        I_smooth = I_smooth_1.to(device)
        I_sharp = I_sharp_1.to(device)

        psd_smooth = compute_psd(I_smooth, device)
        psd_sharp = compute_psd(I_sharp, device)

        I_smooth_fft = compute_fft(I_smooth, device)
        I_sharp_fft = compute_fft(I_sharp, device)
        real_s2sh, real_sh2s = compute_real_filter(I_smooth_fft, I_sharp_fft)
        real_s2sh_smooth = gaussian_blur_2d(real_s2sh)
        real_sh2s_smooth = gaussian_blur_2d(real_sh2s)

        pred_s2sh, pred_sh2s = model(psd_smooth, psd_sharp)

        ft_loss = l1_loss(pred_s2sh, real_s2sh_smooth) + l1_loss(pred_sh2s, real_sh2s_smooth)

        I_gen_sharp, I_gen_smooth = generate_images(
            I_smooth, I_sharp, pred_s2sh, pred_sh2s, device
        )
        recon_loss = (l1_loss(I_gen_sharp, I_sharp) + l1_loss(I_gen_smooth, I_smooth)) / 2.0

        loss = ft_loss + recon_loss

        total_loss += loss.item()
        total_ft_loss += ft_loss.item()
        total_recon_loss += recon_loss.item()
        n_batches += 1

        # Save last batch for visualization
        vis_data = {
            'I_smooth': I_smooth,
            'I_sharp': I_sharp,
            'I_gen_sharp': I_gen_sharp,
            'I_gen_smooth': I_gen_smooth,
            'pred_s2sh': pred_s2sh,
            'pred_sh2s': pred_sh2s,
            'real_s2sh': real_s2sh_smooth,
            'real_sh2s': real_sh2s_smooth,
        }

    return {
        'total_loss': total_loss / n_batches,
        'ft_loss': total_ft_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
    }, vis_data


def main():
    # Config
    IMAGE_ROOT = r"D:\Charan work file\KernelEstimator\Data_Root"
    LR = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 8

    out_dir = Path("training_filter_model")
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "visualization"
    for d in [out_dir, ckpt_dir, vis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | LR: {LR} | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")

    # Dataset
    dataset = PSDDataset(root_dir=IMAGE_ROOT, preload=True)
    n_train = int(0.9 * len(dataset))
    train_set, val_set = random_split(
        dataset, [n_train, len(dataset) - n_train],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}")

        
    model = FilterEstimator().to(device)
    logger.info("Using FilterEstimator (2D)")

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    metrics_log = []
    best_val = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_stats, vis_data = validate(model, val_loader, device)

        scheduler.step(val_stats['total_loss'])

        logger.info(f"Epoch {epoch}/{EPOCHS}")
        logger.info(f"  Train - total: {train_stats['total_loss']:.4f}, ft: {train_stats['ft_loss']:.4f}, recon: {train_stats['recon_loss']:.4f}")
        logger.info(f"  Val   - total: {val_stats['total_loss']:.4f}, ft: {val_stats['ft_loss']:.4f}, recon: {val_stats['recon_loss']:.4f}")

        metrics_log.append({
            'epoch': epoch,
            'train': train_stats,
            'val': val_stats,
        })

        # Save metrics
        with open(out_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_log, f, indent=2)

        # Visualization
        if vis_data:
            plot_filters(
                vis_data['pred_s2sh'], vis_data['pred_sh2s'],
                vis_data['real_s2sh'], vis_data['real_sh2s'],
                epoch, vis_dir
            )
            plot_images(
                vis_data['I_smooth'], vis_data['I_sharp'],
                vis_data['I_gen_sharp'], vis_data['I_gen_smooth'],
                epoch, vis_dir
            )

        # Save checkpoint
        is_best = val_stats['total_loss'] < best_val
        if is_best:
            best_val = val_stats['total_loss']
            logger.info(f"  ** New best: {best_val:.6f} **")

        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val,
        }
        torch.save(ckpt, ckpt_dir / f'epoch_{epoch}.pth')
        if is_best:
            torch.save(ckpt, ckpt_dir / 'best.pth')

    logger.info(f"Training complete. Best val: {best_val:.6f}")


if __name__ == "__main__":
    main()
