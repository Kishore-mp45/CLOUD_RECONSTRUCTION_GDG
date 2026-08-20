import sys
from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure we can import from src
sys.path.append(str(Path(".").resolve()))
from src.cloudremoval.evaluation.visualizer import render_s2_rgb

s2_orig = r"allclear_dataset\roi100663\2022_8\s2_toa\roi100663_s2_toa_2022_8_30_median.tif"
recon = r"outputs\audit_test\audit_test_reconstructed.tif"

def main():
    with rasterio.open(s2_orig) as src:
        orig_arr = src.read()
    with rasterio.open(recon) as src:
        recon_arr = src.read()
        
    recon_rgb = render_s2_rgb(recon_arr)
    orig_rgb = render_s2_rgb(orig_arr)
    
    # cloud mask using Blue band from original
    cloud_mask = orig_arr[1] > 2500
    clear_mask = ~cloud_mask
    
    if np.any(cloud_mask) and np.any(clear_mask):
        print("Clouds found, applying HSV color match...")
        
        # Convert to HSV (V preserves the structure/luminance)
        recon_hsv = mcolors.rgb_to_hsv(recon_rgb)
        
        # We will match H and S
        # But H is circular (0 to 1), mean/std is tricky. 
        # Actually, matching in RGB space but normalizing luminance is also an option.
        # Let's try simple mean/std matching on S and V. Wait, matching H and S?
        # A simpler approach: use Chromaticity (R/(R+G+B), G/(R+G+B), B/(R+G+B))
        # Match chromaticity means/stds, then multiply back by original brightness.
        
        # 1. Calculate Brightness (sum of RGB)
        eps = 1e-6
        recon_sum = np.sum(recon_rgb, axis=-1, keepdims=True) + eps
        
        # 2. Chromaticity
        r_chroma = recon_rgb[:, :, 0] / recon_sum[:, :, 0]
        g_chroma = recon_rgb[:, :, 1] / recon_sum[:, :, 0]
        b_chroma = recon_rgb[:, :, 2] / recon_sum[:, :, 0]
        
        # 3. Match Chromaticity distribution
        for chroma in [r_chroma, g_chroma, b_chroma]:
            c_mean_clear = np.mean(chroma[clear_mask])
            c_std_clear = np.std(chroma[clear_mask])
            c_mean_cloudy = np.mean(chroma[cloud_mask])
            c_std_cloudy = np.std(chroma[cloud_mask])
            
            # Apply shift
            matched = (chroma[cloud_mask] - c_mean_cloudy) * (c_std_clear / max(c_std_cloudy, 1e-5)) + c_mean_clear
            chroma[cloud_mask] = np.clip(matched, 0, 1)
            
        # 4. Re-normalize chromaticity to sum to 1
        c_sum = r_chroma + g_chroma + b_chroma + eps
        r_chroma /= c_sum
        g_chroma /= c_sum
        b_chroma /= c_sum
        
        # 5. Multiply back by original brightness
        matched_rgb = np.zeros_like(recon_rgb)
        matched_rgb[:, :, 0] = r_chroma * recon_sum[:, :, 0]
        matched_rgb[:, :, 1] = g_chroma * recon_sum[:, :, 0]
        matched_rgb[:, :, 2] = b_chroma * recon_sum[:, :, 0]
        
        matched_rgb = np.clip(matched_rgb, 0.0, 1.0)
        
        # Save output
        plt.imsave("outputs/audit_test/color_matched_reconstructed.png", matched_rgb)
        
        # Calculate new ratios
        recon_r_cloudy = np.mean(matched_rgb[:, :, 0][cloud_mask])
        recon_g_cloudy = np.mean(matched_rgb[:, :, 1][cloud_mask])
        recon_b_cloudy = np.mean(matched_rgb[:, :, 2][cloud_mask])
        sum_rgb = recon_r_cloudy + recon_g_cloudy + recon_b_cloudy
        
        print(f"New Green Ratio under clouds: {recon_g_cloudy/sum_rgb:.3f}")
        print("Saved to outputs/audit_test/color_matched_reconstructed.png")
    else:
        print("No clouds or no clear sky.")

if __name__ == "__main__":
    main()
