# Checklist: Data Pile

This document is a one-page checklist to go through, questions to ask, and clarifications to make after you have access to some clinical imaging data, but, before you start to use it to build models.  

- [ ] (1) List out pixel/voxel dimensions (number of elements) and spacing (in mm), for each subject in the data set. See https://simpleitk.readthedocs.io/en/master/link_DicomImagePrintTags_docs.html for more. Saving this meta-data in a .csv file for all the subjects is highly recommended. 

- [ ] (2) Build a reproducible pipeline (fancy word for set of scripts) to convert this raw data into processed/cleaned data in a different image format (e.g., individual 2D DICOM slices to NIfTI in case of 3D volumes). See https://drivendata.github.io/cookiecutter-data-science/#directory-structure for how to structure folders to hold raw, interim, and processed data separately. 

- [ ] (3) Depending on the modality/anatomy, consider reorienting/resampling everything to a standard space: LPS or RAS for example. Medical data includes metadata like coordinate origin and so on which is richer than natural images in computer vision. See https://www.slicer.org/wiki/Coordinate_systems#Anatomical_coordinate_system for more. Pay attention that label data resampling cannot use linear/bicubic interpolation (they create new label categories): use nearest neighbour instead 

- [ ] (4) Where it corresponds, confirm that the data you have is properly anonymized. If you see any personally identifiable information, be sure to inform your supervisor to find out what action to take next. https://pydicom.github.io/pydicom/dev/auto_examples/metadata_processing/plot_anonymize.html and https://www.mathworks.com/help/images/ref/dicomanon.html are examples of how to anonymize DICOM files. 

- [ ] (5) Plot and display data to visually verify that the categories are correct (classification problems), and label masks are mapped to the right anatomy (segmentation problems). Verify for motion artefacts, incorrect volume of interest (e.g., missing anatomical information), etc. https://www.slicer.org, http://www.itksnap.org/pmwiki/pmwiki.php and https://www.mathworks.com/help/images/explore-3-d-volumetric-data-with-volume-viewer-app.html are useful applications for this task. If you see issues, challenge your clinical collaborator or the data provider ;) 

- [ ] (6) (optional) If your data comes from multiple sources, check if there are biases due to acquisition/processing protocols or imaging hardware vendors. Often, these biases could creep into models which can eventually lead to them learning these biases and hence generalizing worse on new data samples.  

- [ ] (7) (optional) Consider registering all the subjects to a known atlas for that anatomy. For brain imaging studies, https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases lists some popular ones. This can be useful as further spatial normalization.  
 

For questions/suggestions for improvements, please create an issue in this repository.
