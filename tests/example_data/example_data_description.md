# MassiPipe example data description

## massimal_larvik_olbergholmen_202308301001-test_hsi
This dataset consists of 2 images acquired during a campaign at Ølbergholmen close to Larvik. Both images were originally 2000 lines long, and approximately 1 GB in size. The images have been "spatially cropped" (in Spectronon) to only include the first 25 lines, reducing the size of each image to approx 13,5 MB. Reducing the file size makes the files more managable in the GitHub repository, and also makes tests run faster, with minimal loss of "representativeness". The corresponding *.times files have also been cropped to the first 25 lines. The data in the *.lcf files are interpolated to the timestamps in the *.times files, and have therefore been left unchanged. 

Cropping the images in Spectronon introduces some changes in the header file; "lines" is updated, "header offset" is removed, formatting of "wavelengths" is slightly changed, and "label" and "history" fields are introduced. These changes have been reverted in the example data, so that the only difference in the header file relative to the original (full) image is that "lines = 25" rather than "lines = 2000".

The calibration files in the "calibration" folder are the most up-to-date calibration files that were available at the time of the campaign. 