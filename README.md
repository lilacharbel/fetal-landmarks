# Undergraduate Final Project: Fetal Brain Linear Measurements

Undergraduate final project conducted at the Sagol School of Neuroscience,
Tel Aviv University. 
The project focuses on developing a semi-automatic method for extracting three linear measurements of the 
fetal brain from 3D MRI scans. The measurements of interest are the Trans Cerebellum Diameter (TCD), 
Bone Biparietal Diameter (BBD), and Cerebral Biparietal Diameter (CBD). 

The project utilizes a modified [HRNet](https://arxiv.org/abs/1908.07919) architecture, 
which incorporates two heads: a classification head for reference slice detection and
a landmarks detection head.

<img src="figures/TCD.png" alt="TCD" title="TCD" height=250> <img src="figures/BBD.png" alt="BBD" title="BBD" height=250> <img src="figures/CBD.png" alt="CBD" title="CBD" height=250>

The model takes 3D MRI scans of fetal brains as inputs, along with brain segmentation data.
The brain segmentation process identifies the region of interest (ROI) in each scan.
The output is the 2 landmarks locations on the selected reference slice for the relevant measurement.