import sys
from pathlib import Path
import rasterio
import numpy as np

sys.path.append(str(Path(".").resolve()))
from src.cloudremoval.evaluation.visualizer import render_s2_rgb, apply_chromaticity_match, RGB_INDICES

def main():
    recon_path = r"outputs\inference\inf_20260820_074016_8c136b_reconstructed.tif"
    s2_path = r"allclear_dataset\roi502413\2022_10\s2_toa\roi502413_s2_toa_2022_10_19_median.tif"
    
    with rasterio.open(recon_path) as src:
        recon_arr = src.read()
    
    with rasterio.open(s2_path) as src:
        orig_arr = src.read()
        
    print(f"Recon array shape: {recon_arr.shape}")
    print(f"Orig array shape: {orig_arr.shape}")
    
    rgb = render_s2_rgb(recon_arr, rgb_indices=RGB_INDICES)
    print(f"Rendered RGB shape: {rgb.shape}, min: {rgb.min()}, max: {rgb.max()}")
    
    # Let's do exactly what apply_chromaticity_match does and print stats
    orig = orig_arr
    cloud_threshold = 2500.0
    cloud_mask = orig[1] > cloud_threshold
    clear_mask = ~cloud_mask
    
    print(f"Cloud pixels: {np.sum(cloud_mask)} ({np.mean(cloud_mask)*100:.1f}%)")
    print(f"Clear pixels: {np.sum(clear_mask)} ({np.mean(clear_mask)*100:.1f}%)")
    
    rgb_matched = apply_chromaticity_match(rgb, orig_arr)
    print(f"Matched RGB shape: {rgb_matched.shape}, min: {rgb_matched.min()}, max: {rgb_matched.max()}, any nans? {np.any(np.isnan(rgb_matched))}")
    
    import matplotlib.pyplot as plt
    plt.imsave("outputs/inference/debug_chroma.png", rgb_matched)

if __name__ == "__main__":
    main()
