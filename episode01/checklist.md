# Checklist: Data Pile

This document is a one-page checklist with: questions-to-ask-yourself *after* you have access to clinical imaging data, but, *before* you start to use it to build models.  

- [ ] (1) List out pixel/voxel dimensions (number of elements) and spacing (in mm), for each subject in the data set (See [here](https://simpleitk.readthedocs.io/en/master/link_DicomImagePrintTags_docs.html) for how to list them from DICOM). We highly recommend saving this meta-data in a .csv file for each subject. 

- [ ] (2) Build a reproducible pipeline 🛠 to convert this raw data into processed/cleaned data in a different image format (e.g., individual 2D DICOM slices to NIfTI in case of 3D volumes). We highly recommend using [cookiecutter](https://drivendata.github.io/cookiecutter-data-science/#directory-structure) to structure your project: especially separating raw, interim, and processed data to avoid corruption. 

- [ ] (3) Depending on the modality/anatomy, consider reorienting/resampling everything to a standard space: LPS or RAS for example. Medical data includes metadata like coordinate origin and so on which is richer than natural images in computer vision. See [here](https://www.slicer.org/wiki/Coordinate_systems#Anatomical_coordinate_system) for more. Achtung 💥 : label data resampling cannot use linear/bicubic interpolation (they create new label categories): consider using nearest neighbour instead.

- [ ] (4) Wherever necessary, confirm that the data you have is properly anonymized (🦸 Batman, instead of Bruce Wayne; Spiderman, instead of Peter Parker, you get the drift). If you see any personally identifiable information, be sure to inform your supervisor to find out what action to take next. Here are some tools using [pydicom](https://pydicom.github.io/pydicom/dev/auto_examples/metadata_processing/plot_anonymize.html), [deid](https://github.com/pydicom/deid) and [MATLAB](https://www.mathworks.com/help/images/ref/dicomanon.html) to anonymize DICOM files. 

- [ ] (5) Plot data to visually check that the categories are correct (classification problems), and label masks are mapped to the right anatomy (segmentation problems). Verify for motion artefacts, incorrect volume of interest (e.g., missing anatomical information), etc. We recommend using [3D Slicer](https://www.slicer.org), [ITKSnap](http://www.itksnap.org/pmwiki/pmwiki.php) and [MATLAB Volume Viewer](https://www.mathworks.com/help/images/explore-3-d-volumetric-data-with-volume-viewer-app.html). Also check for duplicates in the image data ([Here's a tool](https://github.com/elisemercury/Duplicate-Image-Finder) to automate this). If you see issues, challenge your clinical collaborator or the data provider 😉 (see Episode 02 for more!) 

- [ ] (6) (optional) If your data comes from multiple sources, check if there are biases due to acquisition/processing protocols or imaging hardware vendors. Often, these biases could creep into models which can eventually lead to them learning these biases and hence generalizing worse on new data samples. Your clinical collaborator or the image metadata should have this information.

- [ ] (7) (optional) Consider registering all the subjects to a known atlas for that anatomy. For brain imaging studies, [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases) lists some popular ones. This can be useful as further spatial normalization.  
 

For questions/suggestions for improvements, please [create an issue](https://github.com/ubern-mia/bender/issues) in this repository.
