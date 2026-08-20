import sys
from pathlib import Path
import rasterio

# Ensure we can import from src
sys.path.append(str(Path(".").resolve()))
from src.cloudremoval.evaluation.visualizer import render_s2_rgb, apply_chromaticity_match, RGB_INDICES

def main():
    print("Testing visualizer functions...")
    
    s2_orig_path = "allclear_dataset/roi100663/2022_8/s2_toa/roi100663_s2_toa_2022_8_30_median.tif"
    recon_path = "outputs/audit_test/audit_test_reconstructed.tif"
    
    with rasterio.open(recon_path) as src:
        recon_arr = src.read()
        
    print("Rendering recon...")
    rgb = render_s2_rgb(recon_arr, rgb_indices=RGB_INDICES)
    
    print("Reading orig...")
    with rasterio.open(s2_orig_path) as src:
        orig_arr = src.read()
        
    print("Applying chromaticity match...")
    rgb = apply_chromaticity_match(rgb, orig_arr)
    
    print("Saving...")
    import matplotlib.pyplot as plt
    plt.imsave("outputs/audit_test/api_test_matched.png", rgb)
    
    print("SUCCESS")

if __name__ == "__main__":
    main()
