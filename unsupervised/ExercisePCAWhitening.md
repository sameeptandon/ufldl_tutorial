---
layout: post
title:  "Exercise: PCA Whitening"
date:   2013-08-28 13:32:12
categories: unsupervised
---
### PCA and Whitening on natural images ### 

In this exercise, you will implement PCA, PCA whitening and ZCA whitening, and apply them to image patches taken from natural images. 

You will build on the MATLAB starter code which we have provided
in the [Github repository](https://github.com/amaas/stanford_dl_ex) You need only write code at the places
indicated by `YOUR CODE HERE` in the files. The only file you need
to modify is `pca_gen.m`.

### Step 0: Prepare data ###
#### Step 0a: Load data ####

The starter code contains code to load a set of MNIST images. The raw patches will look something like this:

<img src="{{site.baseurl}}/images/Raw_images.png" width="100%">

These patches are stored as column vectors <m>x^{(i)} \in \mathbb{R}^{144}</m> in the <m>144 \times 10000</m> matrix <m>x</m>.

#### Step 0b: Zero mean the data ####

First, for each image patch, compute the mean pixel value and subtract it from that image, this centering the image around zero.  You should compute a different mean value for each image patch.

### Step 1: Implement PCA ###

#### Step 1a: Implement PCA ####

In this step, you will implement PCA to obtain <m>x_{\rm rot}</m>, the matrix in which the data is "rotated" to the basis comprising the principal components (i.e. the eigenvectors of <m>\Sigma</m>). Note that in this part of the exercise, you should ''not'' whiten the data.

#### Step 1b: Check covariance ####

To verify that your implementation of PCA is correct, you should check the covariance matrix for the rotated data <m>x_{\rm rot}</m>.  PCA guarantees that the covariance matrix for the rotated data is a diagonal matrix (a matrix with non-zero entries only along the main diagonal). Implement code to compute the covariance matrix and verify this property. One way to do this is to compute the covariance matrix, and visualise it using the MATLAB command `imagesc`. The image should show a coloured diagonal line against a blue background. For this dataset, because of the range of the diagonal entries, the diagonal line may not be apparent, so you might get a figure like the one show below, but this trick of visualizing using `imagesc` will come in handy later in this exercise. 

<img src="{{site.baseurl}}/images/Pca_covar.png" width="100%">

### Step 2: Find number of components to retain ###

Next, choose <m>k</m>, the number of principal components to retain.  Pick <m>k</m> to be as small as possible, but so that at least 99% of the variance is retained.  In the step after this, you will discard all but the top <m>k</m> principal components, reducing the dimension of the original data to <m>k</m>.

### Step 3: PCA with dimension reduction ###

Now that you have found <m>k</m>, compute <m>\tilde{x}</m>, the reduced-dimension representation of the data.  This gives you a representation of each image patch as a <m>k</m> dimensional vector instead of a 144 dimensional vector.  If you are training a sparse autoencoder or other algorithm on this reduced-dimensional data, it will run faster than if you were training on the original 144 dimensional data. 

To see the effect of dimension reduction, go back from <m>\tilde{x}</m> to produce the matrix <m>\hat{x}</m>, the dimension-reduced data but expressed in the original 144 dimensional space of image patches. Visualise <m>\hat{x}</m> and compare it to the raw data, <m>x</m>. You will observe that there is little loss due to throwing away the principal components that correspond to dimensions with low variation. For comparison, you may also wish to generate and visualise <m>\hat{x}</m> for when only 90% of the variance is retained.  

<table>
<tr>
<td> <img src="{{site.baseurl}}/images/Raw_images.png" width="100%"> </td>
<td> <img src="{{site.baseurl}}/images/Pca_images.png" width="100%"> </td> 
<td>
<img src="{{site.baseurl}}/images/Pca_images_90.png" width="100%"> 
</td>
</tr>
<tr>
<td>Raw images <br /> &nbsp; </td>
<td>PCA dimension-reduced images<br /> (99% variance)</td>
<td>PCA dimension-reduced images<br /> (90% variance)</td>
</tr>
</table>

### Step 4: PCA with whitening and regularization ###

#### Step 4a: Implement PCA with whitening and regularization ####

Now implement PCA with whitening and regularization to produce the matrix <m>x_{PCAWhite}</m>.  Use the following parameter value:

 `epsilon = 0.1`

#### Step 4b: Check covariance ####

 Similar to using PCA alone, PCA with whitening also results in processed data that has a diagonal covariance matrix. However, unlike PCA alone, whitening additionally ensures that the diagonal entries are equal to 1, i.e. that the covariance matrix is the identity matrix. 

 That would be the case if you were doing whitening alone with no regularization. However, in this case you are whitening with regularization, to avoid numerical/etc. problems associated with small eigenvalues.  As a result of this, some of the diagonal entries of the covariance of your <m>x_{\rm PCAwhite}</m> will be smaller than 1.  

 To verify that your implementation of PCA whitening with and without regularization is correct, you can check these properties. Implement code to compute the covariance matrix and verify this property. (To check the result of PCA without whitening, simply set epsilon to 0, or close to 0, say `1e-10`).  As earlier, you can visualise the covariance matrix with `imagesc`. When visualised as an image, for PCA whitening without regularization you should see a red line across the diagonal (corresponding to the one entries) against a blue background (corresponding to the zero entries); for PCA whitening with regularization you should see a red line that slowly turns blue across the diagonal (corresponding to the 1 entries slowly becoming smaller). 

 <table>
 <tr>
 <td>
 <img src="{{site.baseurl}}/images/Pca_whitened_covar.png" width="100%">
 </td>
 <td>
 <img src="{{site.baseurl}}/images/Pca_whitened_unregularised_covar.png" width="100%">
 </td>
 </tr>
 <tr>
 <td><center>Covariance for PCA whitening with regularization</center></td>
 <td><center>Covariance for PCA whitening without regularization</center></td>
 </tr>
 </table>

### Step 5: ZCA whitening ###

 Now implement ZCA whitening to produce the matrix <m>x_{ZCAWhite}</m>. Visualize <m>x_{ZCAWhite}</m> and compare it to the raw data, <m>x</m>. You should observe that whitening results in, among other things, enhanced edges.  Try repeating this with `epsilon` set to 1, 0.1, and 0.01, and see what you obtain.  The example shown below (left image) was obtained with `epsilon = 0.1`. 

 <table>
 <tr>
 <td>

 <img src="{{site.baseurl}}/images/Zca_whitened_images.png" width="100%">
 </td><td>
 <img src="{{site.baseurl}}/images/Raw_images.png" width="100%">
 </td>
 </tr>
 <tr>
 <td>ZCA whitened images</td>
 <td>Raw images</td>
 </tr>
 </table>
