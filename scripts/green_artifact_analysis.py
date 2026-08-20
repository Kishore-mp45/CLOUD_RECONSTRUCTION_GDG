import rasterio
import numpy as np

# Use the audit test output from the previous run
s2_orig = r"allclear_dataset\roi100663\2022_8\s2_toa\roi100663_s2_toa_2022_8_30_median.tif"
recon = r"outputs\audit_test\audit_test_reconstructed.tif"

def analyze_green_bias():
    with rasterio.open(s2_orig) as src:
        orig_data = src.read()
    with rasterio.open(recon) as src:
        recon_data = src.read()

    # B4 (Red) is idx 3, B3 (Green) is idx 2, B2 (Blue) is idx 1
    orig_b4, orig_b3, orig_b2 = orig_data[3], orig_data[2], orig_data[1]
    recon_b4, recon_b3, recon_b2 = recon_data[3], recon_data[2], recon_data[1]

    # Simple cloud mask: very bright in original B2 (Blue/Aerosol usually high for clouds)
    cloud_mask = orig_b2 > 2500

    print("=== ANALYSIS OF RECONSTRUCTED PIXELS ===")
    
    if np.any(cloud_mask):
        recon_r_cloudy = np.mean(recon_b4[cloud_mask])
        recon_g_cloudy = np.mean(recon_b3[cloud_mask])
        recon_b_cloudy = np.mean(recon_b2[cloud_mask])
        
        print(f"Mean Reconstructed RGB under THICK CLOUDS:")
        print(f"  R: {recon_r_cloudy:.1f}")
        print(f"  G: {recon_g_cloudy:.1f}")
        print(f"  B: {recon_b_cloudy:.1f}")
        
        # Calculate Green Ratio
        sum_rgb_cloudy = recon_r_cloudy + recon_g_cloudy + recon_b_cloudy
        if sum_rgb_cloudy > 0:
            print(f"  Green Ratio: {recon_g_cloudy / sum_rgb_cloudy:.3f} (Values > 0.33 indicate green bias)")
            print(f"  Blue Ratio:  {recon_b_cloudy / sum_rgb_cloudy:.3f}")
            print(f"  Red Ratio:   {recon_r_cloudy / sum_rgb_cloudy:.3f}")
    else:
        print("No thick clouds found in this scene.")

    clear_mask = ~cloud_mask
    if np.any(clear_mask):
        recon_r_clear = np.mean(recon_b4[clear_mask])
        recon_g_clear = np.mean(recon_b3[clear_mask])
        recon_b_clear = np.mean(recon_b2[clear_mask])
        
        print(f"\nMean Reconstructed RGB under CLEAR SKY (Passthrough):")
        print(f"  R: {recon_r_clear:.1f}")
        print(f"  G: {recon_g_clear:.1f}")
        print(f"  B: {recon_b_clear:.1f}")
        
        # Calculate Green Ratio
        sum_rgb_clear = recon_r_clear + recon_g_clear + recon_b_clear
        if sum_rgb_clear > 0:
            print(f"  Green Ratio: {recon_g_clear / sum_rgb_clear:.3f}")
            print(f"  Blue Ratio:  {recon_b_clear / sum_rgb_clear:.3f}")
            print(f"  Red Ratio:   {recon_r_clear / sum_rgb_clear:.3f}")

if __name__ == "__main__":
    analyze_green_bias()
