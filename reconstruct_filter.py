import numpy as np
import nibabel as nib
import torch
import os
from pathlib import Path
from FilterModel import FilterEstimator
from TestDataset import TestDataset
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(checkpoint_path, use_light=True):
    model = FilterEstimator()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'Loaded model from {checkpoint_path}')
    return model


def compute_psd(img_tensor):
    """Compute normalized PSD from image tensor."""
    with torch.no_grad():
        if img_tensor.dim() == 4:
            x = img_tensor[:, 0, :, :]
        else:
            x = img_tensor
        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
        psd = torch.abs(fft_shifted) ** 2
        psd = torch.log(psd + 1)

        B = psd.shape[0] if psd.dim() == 3 else 1
        if psd.dim() == 2:
            psd = psd.unsqueeze(0)

        psd_flat = psd.view(B, -1)
        psd_min = psd_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
        psd_max = psd_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)
        psd = (psd - psd_min) / (psd_max - psd_min + 1e-10)

        return psd.unsqueeze(1)  # [B, 1, H, W]


def compute_fft(image):
    """Compute shifted FFT."""
    if image.dim() == 4:
        image = image[:, 0, :, :]
    fft = torch.fft.fft2(image)
    return torch.fft.fftshift(fft, dim=(-2, -1))


def apply_filter(image_fft, filter_2d):
    """Apply filter in frequency domain and return spatial image."""
    filtered_fft = image_fft * filter_2d
    filtered_fft = torch.fft.ifftshift(filtered_fft, dim=(-2, -1))
    result = torch.fft.ifft2(filtered_fft).real
    return torch.clamp(result, 0, 1)


def extract_kernel_name(filename):
    if '_filter_' in filename:
        return filename.split('_filter_')[1].split('.')[0]
    return 'unknown'


def reconstruct_volume(sample, model, output_dir, save_filters=True):
    """Reconstruct a volume and optionally save the filter ratios."""
    data_smooth = sample['smooth_volume']
    data_sharp = sample['sharp_volume']
    volume_id = sample['volume_id']
    smooth_kernel = extract_kernel_name(sample['smooth_file'])
    sharp_kernel = extract_kernel_name(sample['sharp_file'])
    num_slices = data_smooth.shape[2]

    vol_generated_sharp = np.zeros_like(data_smooth, dtype=np.float32)
    vol_generated_smooth = np.zeros_like(data_sharp, dtype=np.float32)

    # Store filter profiles for analysis
    filter_profiles_s2sh = []
    filter_profiles_sh2s = []

    print(f"Processing {volume_id}: {num_slices} slices")

    for k in range(num_slices):
        # Get slices
        s_slice = data_smooth[:, :, k].copy()
        h_slice = data_sharp[:, :, k].copy()

        # Normalize
        s_slice = np.clip(s_slice, -1000, 3000)
        h_slice = np.clip(h_slice, -1000, 3000)
        s_slice_norm = (s_slice + 1000) / 4000
        h_slice_norm = (h_slice + 1000) / 4000

        # To tensor
        I_smooth = torch.from_numpy(s_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)
        I_sharp = torch.from_numpy(h_slice_norm).float().unsqueeze(0).unsqueeze(0).to(device)

        # Compute PSDs
        psd_smooth = compute_psd(I_smooth)
        psd_sharp = compute_psd(I_sharp)

        with torch.no_grad():
            # Get filter predictions
            filter_s2sh, filter_sh2s = model(psd_smooth, psd_sharp)

            # Apply filters
            I_smooth_fft = compute_fft(I_smooth)
            I_sharp_fft = compute_fft(I_sharp)

            I_gen_sharp = apply_filter(I_smooth_fft, filter_s2sh)
            I_gen_smooth = apply_filter(I_sharp_fft, filter_sh2s)

        # Store filter profiles (center row)
        center = filter_s2sh.shape[-1] // 2
        filter_profiles_s2sh.append(filter_s2sh[0, center, :].cpu().numpy())
        filter_profiles_sh2s.append(filter_sh2s[0, center, :].cpu().numpy())

        # Convert back to HU
        res_sharp = I_gen_sharp.cpu().numpy().squeeze()
        res_smooth = I_gen_smooth.cpu().numpy().squeeze()

        vol_generated_sharp[:, :, k] = (res_sharp * 4000) - 1000
        vol_generated_smooth[:, :, k] = (res_smooth * 4000) - 1000

    # Clip to valid HU range
    vol_generated_sharp = np.clip(vol_generated_sharp, -1000, 3000)
    vol_generated_smooth = np.clip(vol_generated_smooth, -1000, 3000)

    # Save reconstructed volumes
    nii_sharp = nib.Nifti1Image(vol_generated_sharp, sample['sharp_affine'], sample['sharp_header'])
    nii_smooth = nib.Nifti1Image(vol_generated_smooth, sample['smooth_affine'], sample['smooth_header'])

    sharp_path = output_dir / f'{volume_id}_{smooth_kernel}_to_{sharp_kernel}.nii.gz'
    smooth_path = output_dir / f'{volume_id}_{sharp_kernel}_to_{smooth_kernel}.nii.gz'

    nib.save(nii_sharp, str(sharp_path))
    nib.save(nii_smooth, str(smooth_path))
    print(f'Saved: {sharp_path.name}, {smooth_path.name}')

    # Save filter profiles
    if save_filters:
        filter_dir = output_dir / 'filters'
        filter_dir.mkdir(exist_ok=True)

        # Average filter across slices
        avg_s2sh = np.mean(filter_profiles_s2sh, axis=0)
        avg_sh2s = np.mean(filter_profiles_sh2s, axis=0)

        # Save as numpy
        np.save(filter_dir / f'{volume_id}_filter_s2sh.npy', avg_s2sh)
        np.save(filter_dir / f'{volume_id}_filter_sh2s.npy', avg_sh2s)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(avg_s2sh)
        axes[0].set_title(f'Filter S2SH ({smooth_kernel} → {sharp_kernel})')
        axes[0].set_xlabel('Frequency bin')
        axes[0].set_ylabel('Magnitude')
        axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        axes[1].plot(avg_sh2s)
        axes[1].set_title(f'Filter SH2S ({sharp_kernel} → {smooth_kernel})')
        axes[1].set_xlabel('Frequency bin')
        axes[1].set_ylabel('Magnitude')
        axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(filter_dir / f'{volume_id}_filters.png', dpi=150)
        plt.close()
        print(f'Saved filter profiles to {filter_dir}')

    return vol_generated_sharp, vol_generated_smooth


def main():
    # Config
    checkpoint_path = r"D:\Charan work file\PhantomTesting\Code\training_filter_model\checkpoints\best.pth"
    data_root = r"D:\Charan work file\KernelEstimator\Data_Root"
    output_dir = Path(r"D:\Charan work file\PhantomTesting\reconstructions_filter_model")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path, use_light=True)

    # Load dataset
    dataset = TestDataset(root_dir=data_root, preload=True)
    print(f'Loaded {len(dataset)} volumes')

    # Process all volumes
    for idx in range(len(dataset)):
        print(f'\nProcessing volume {idx+1}/{len(dataset)}')
        sample = dataset[idx]
        reconstruct_volume(sample, model, output_dir, save_filters=True)

    print(f'\nDone! Results saved to {output_dir}')


if __name__ == "__main__":
    main()