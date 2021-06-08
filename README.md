# COV19 Detection on CT Scans

The project aims to analyze computed tomography (CT) images and classify them into one of three classes: COV, Normal, OtherPneumonia. The dataset is publicly available under [this link](https://github.com/m2dgithub/CT-COV19).

![one example from each class](images/classes.png)

The project consists of Jupyter notebooks and additional py files with model architectures and some useful functions.

## Initial analysis and preprocessing

The analysis of the dataset and other visualisations can be found in:

* [Analysis and Visualisations](notebooks/Analysis-Visualisations.ipynb) - analysis of the dataset
* [Data Augmentation](notebooks/Data-Augmentation.ipynb) - images after data augmentation

Preprocessing steps are presented in:

* [Patient Segmentation](notebooks/Background-Removal.ipynb) - finding patient's contour and background removal
![original image, patient contour and image with removed background](images/body&#32;contour.png)

* [Lungs Segmentation](notebooks/Skimage-Lungs-Segmentation.ipynb) - lungs segmentation using skimage segmentation
![lungs contour and segmented lungs](images/skimage&#32;active&#32;contour.png)

## Results

Metrics and results plots are spread over several notebooks:

* [F1 Scores](notebooks/F1-Scores.ipynb) - f1 scores achieved by models on validation set
* [Confusion Matrices](notebooks/Confusion-Matrices.ipynb) - confusion matrices for each model configuration
* [AUC Scores](notebooks/AUC-Scores.ipynb) - AUC scores for all models configurations along with ROC and Precision-Recall curves for the best models

Explainability of the solution is taken up here:

* [GradCAM](notebooks/GradCAM.ipynb) - GradCAM analysis

## Disclaimer

This project is intended for educational purposes only. It is not a substitute for professional medical advice, diagnosis or treatment. 
