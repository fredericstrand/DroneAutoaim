
Drone - v1 2023-08-20 9:22am
==============================

This dataset was exported via roboflow.com on August 21, 2023 at 3:34 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 17706 images.
Drone are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* Random rotation of between -34 and +34 degrees
* Random shear of between -34° to +34° horizontally and -9° to +9° vertically
* Random brigthness adjustment of between -38 and +38 percent
* Random Gaussian blur of between 0 and 3 pixels
* Salt and pepper noise was applied to 14 percent of pixels


