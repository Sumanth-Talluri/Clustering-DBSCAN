# Clustering-DBSCAN

## Purpose

In this project I am applying clustering techniques to detect the number of clusters in the extracted brain slices of resting state functional magnetic resonance imaging (rs-fMRI) scans.


## Objectives

* To perform cluster detection in the brain slices.

## Description

In this project, the program will take a patient’s dataset, performs brain slice extraction on it and then detect the number of clusters present in every extracted brain slice.

## Tasks

* Extract the brain slices in every image (similar to [Brain Slices Extraction](https://github.com/Sumanth-Talluri/Brain-Slices-Extraction)).
* Once I have the brain slices images, I am applying clustering techniques to detect the number of clusters present in every slice. To extract the noticeable big enough cluster, I only am reporting the number of clusters whose pixel value is greater than 135 pixels.


### Files

* clustering.py - The clustering.py will read all the images (images those end with word “thresh”) from the given data and perform slices extraction. Once I have brain slices images, I will count number of clusters every slice contains using clustering techniques like DBSCAN.


* test.py -  This file is executed and it will call the functions in clustering.py.

* testPatient - The test.py reads a folder named ‘testPatient’ and outputs two folders. One folder named “Slices” and another folder named "Clusters". ‘Slices’ folder will further have ‘N’ number of folders where N is number of images that ends with “thresh”. Folder ‘Clusters’ will also have N number of folders and every folder will have clusters detected images along with one ‘csv’ file which will report the number of clusters for every slice in that image folder.
