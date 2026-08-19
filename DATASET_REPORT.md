# Dataset inspection report

Dataset root inspected: `allclear_dataset`

## Layout

The root contains ROI directories named `roi<id>` (for example `roi11329`).  Each
ROI contains time folders named `<year>_<month>` (for example `2022_6`).  Those
contain two sensor directories:

```
allclear_dataset/
  roi11329/
    2022_6/
      s1/      roi11329_s1_2022_6_14_median.tif
      s2_toa/  roi11329_s2_toa_2022_6_14_median.tif
```

Filenames follow `roi<id>_<sensor>_<year>_<month>_<day>_median.tif`.
The common `(ROI, year, month, day)` key matches S1 and S2 acquisitions.  For
example, `roi11329`, `2022_6`, day `14` has both S1 and S2 files.

## Triplet labels: supplied by metadata

`allclear_test_metadata.json` is the training manifest that supplies the roles
not encoded by the folders. Each sample has a `target` list containing the
cloud-free S2 target, an `s2_toa` list containing three input S2 observations,
and an optional `s1` list. The absolute `/scratch/...` paths in the manifest map
to this checkout by their final `roi/year_month/sensor/filename.tif` components.

There are 3,698 manifest samples; 2,495 have a target, S2 input, and S1 file
present locally. Example key `roi83528_2022-03-13_2022-03-28` maps to S2 input
2022-03-13, target 2022-03-18, and S1 2022-03-24. These dates are close but not
identical, so match samples via the manifest—not by an exact filename date.

## Verified sample metadata

`roi11329/2022_6/s2_toa/roi11329_s2_toa_2022_6_14_median.tif` is 310×311,
13-band `float64`, CRS `EPSG:32760`, with descriptions `B1, B2, ..., B8, B8A,
B9, B10, B11, B12`.  RGB is explicitly B4/B3/B2, at 1-based band indexes 4/3/2.

`roi11329/2022_6/s1/roi11329_s1_2022_6_14_median.tif` is 310×311, 2-band
`float64`, CRS `EPSG:32760`, with descriptions `VV, VH`; its sample values are
negative (VV −26.41 to −6.43; VH −44.39 to −19.74), consistent with dB-like SAR
backscatter values.
