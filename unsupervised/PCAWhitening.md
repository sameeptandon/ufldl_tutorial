---
layout: post
title:  "PCA Whitening"
date:   2013-08-28 13:32:12
categories: unsupervised
---
### Introduction ###
Principal Components Analysis (PCA) is a dimensionality reduction
algorithm that can be used to significantly speed up your
unsupervised feature learning algorithm.  More importantly,
understanding PCA will enable us to later implement **whitening**,
which is an important pre-processing step for many algorithms. 

Suppose you are training your algorithm on images.  Then the input
will be somewhat redundant, because the values of adjacent pixels
in an image are highly correlated.  Concretely, suppose we are
training on 16x16 grayscale image patches.  Then <m>\textstyle x
\in \Re^{256}</m> are 256 dimensional vectors, with one feature <m>\textstyle x_j</m> corresponding to the intensity of each
pixel.  Because of the correlation between adjacent pixels, PCA
will allow us to approximate the input with a much lower
dimensional one, while incurring very little error.

### Example and Mathematical Background ###

For our running example, we will use a dataset  <m>\textstyle \{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}</m> with  <m>\textstyle n=2</m> dimensional inputs, so that  <m>\textstyle x^{(i)} \in \Re^2</m>.
Suppose we want to reduce the data from 2 dimensions to 1.  (In
practice, we might want to reduce data from 256 to 50 dimensions,
say; but using lower dimensional data in our example allows us to
visualize the algorithms better.)  Here is our dataset:

<img src="{{site.baseurl}}/images/PCA-rawdata.png" width="100%">

This data has already been pre-processed so that each of the features <m>\textstyle x_1</m> and <m>\textstyle x_2</m>
have about the same mean (zero) and variance.  

For the purpose of illustration, we have also colored each of the points one of
three colors, depending on their <m>\textstyle x_1</m> value; these colors are not used by the
algorithm, and are for illustration only.

PCA will find a lower-dimensional subspace onto which to project our data.  
From visually examining the data, it appears that <m>\textstyle u_1</m> is the principal direction of 
variation of the data, and <m>\textstyle u_2</m> the secondary direction of variation:

<img src="{{site.baseurl}}/images/PCA-u1.png" width="100%">

I.e., the data varies much more in the direction <m>\textstyle u_1</m> than <m>\textstyle u_2</m>. 
To more formally find the directions <m>\textstyle u_1</m> and <m>\textstyle u_2</m>, we first compute the matrix <m>\textstyle \Sigma</m>
as follows:

<m>\begin{align}
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)})(x^{(i)})^T. 
\end{align}</m>

If <m>\textstyle x</m> has zero mean, then <m>\textstyle \Sigma</m> is exactly the covariance matrix of <m>\textstyle x</m>.  (The symbol "<m>\textstyle \Sigma</m>", pronounced "Sigma", is the standard notation for denoting the covariance matrix.  Unfortunately it looks just like the summation symbol, as in <m>\sum_{i=1}^n i</m>; but these are two different things.) 

It can then be shown that <m>\textstyle u_1</m>---the principal direction of variation of the data---is 
the top (principal) eigenvector of <m>\textstyle \Sigma</m>, and <m>\textstyle u_2</m> is
the second eigenvector.

Note: If you are interested in seeing a more formal mathematical derivation/justification of this result, see the CS229 (Machine Learning) lecture notes on PCA (link at bottom of this page).  You won't need to do so to follow along this course, however.  

You can use standard numerical linear algebra software to find these eigenvectors (see Implementation Notes).
Concretely, let us compute the eigenvectors of <m>\textstyle \Sigma</m>, and stack
the eigenvectors in columns to form the matrix <m>\textstyle U</m>:

<m>\begin{align}
U = 
\begin{bmatrix} 
| &amp; | &amp; &amp; |  \\
u_1 &amp; u_2 &amp; \cdots &amp; u_n  \\
| &amp; | &amp; &amp; | 
\end{bmatrix}       
\end{align}</m>

Here, <m>\textstyle u_1</m> is the principal eigenvector
(corresponding to the largest eigenvalue), <m>\textstyle u_2</m>
is the second eigenvector, and so on.  Also, let <m>\textstyle
\lambda_1, \lambda_2, \ldots, \lambda_n</m> be the corresponding
eigenvalues. 

The vectors <m>\textstyle u_1</m> and <m>\textstyle u_2</m> in our example form a new basis in which we 
can represent the data.  Concretely, let <m>\textstyle x \in \Re^2</m> be some training example.  Then <m>\textstyle u_1^Tx</m>
is the length (magnitude) of the projection of <m>\textstyle x</m> onto the vector <m>\textstyle u_1</m>.  

Similarly, <m>\textstyle u_2^Tx</m> is the magnitude of <m>\textstyle x</m> projected onto the vector <m>\textstyle u_2</m>.

### Rotating the Data ###

Thus, we can represent <m>\textstyle x</m> in the <m>\textstyle (u_1, u_2)</m>-basis by computing

<m>\begin{align}
x_{\rm rot} = U^Tx = \begin{bmatrix} u_1^Tx \\ u_2^Tx \end{bmatrix} 
\end{align}</m>

(The subscript "rot" comes from the observation that this corresponds to
a rotation (and possibly reflection) of the original data.)
Lets take the entire training set, and compute  <m>\textstyle x_{\rm rot}^{(i)} = U^Tx^{(i)}</m> for every <m>\textstyle i</m>.  Plotting this transformed data <m>\textstyle x_{\rm rot}</m>, we get: 

<img src="{{site.baseurl}}/images/PCA-rotated.png" width="100%">

This is the training set rotated into the <m>\textstyle u_1</m>,<m>\textstyle u_2</m> basis. In the general
case, <m>\textstyle U^Tx</m> will be the training set rotated into the basis <m>\textstyle u_1</m>,<m>\textstyle u_2</m>, ...,<m>\textstyle u_n</m>. 

One of the properties of <m>\textstyle U</m> is that it is an "orthogonal" matrix, which means
that it satisfies <m>\textstyle U^TU = UU^T = I</m>. 
So if you ever need to go from the rotated vectors <m>\textstyle x_{\rm rot}</m> back to the 
original data <m>\textstyle x</m>, you can compute 

<m>\begin{align}
x = U x_{\rm rot}   ,
\end{align}</m>

because <m>\textstyle U x_{\rm rot} =  UU^T x = x</m>.

### Reducing the Data Dimension ###

We see that the principal direction of variation of the data is the first
dimension <m>\textstyle x_{\rm rot,1}</m> of this rotated data.  Thus, if we want to
reduce this data to one dimension, we can set 

<m>\begin{align}
\tilde{x}^{(i)} = x_{\rm rot,1}^{(i)} = u_1^Tx^{(i)} \in \Re.
\end{align}</m>

More generally, if <m>\textstyle x \in \Re^n</m> and we want to reduce 
it to a <m>\textstyle k</m> dimensional 
representation <m>\textstyle \tilde{x} \in \Re^k</m> 
(where k < n), 
we would take the first <m>\textstyle k </m>
components of <m>\textstyle x_{\rm rot}</m>, 
which correspond to the top <m>\textstyle k</m> 
directions of variation. 

Another way of explaining PCA is that <m>\textstyle x_{\rm rot}</m> is an <m>\textstyle n</m> dimensional
vector, where the first few components are likely to 
be large (e.g., in our example, we saw that <m>\textstyle x_{\rm rot,1}^{(i)} = u_1^Tx^{(i)}</m> takes
reasonably large values for most examples <m>\textstyle i</m>), and
the later components are likely to be small (e.g., in our example, <m>\textstyle x_{\rm rot,2}^{(i)} = u_2^Tx^{(i)}</m> was more likely to be small).  What
PCA does it it 
drops the the later (smaller) components of <m>\textstyle x_{\rm rot}</m>, and
just approximates them with 0's.  Concretely, our definition of  <m>\textstyle \tilde{x}</m> can also be arrived at by using an approximation to <m>\textstyle x_{\rm rot}</m> where all but the first <m>\textstyle k</m> components are zeros.  In other words, we have: 

<m>\begin{align}
\tilde{x} = 
\begin{bmatrix} 
x_{\rm rot,1} \\
\vdots \\ 
x_{\rm rot,k} \\
0 \\ 
\vdots \\ 
0 \\ 
\end{bmatrix}
\approx 
\begin{bmatrix} 
x_{\rm rot,1} \\
\vdots \\ 
x_{\rm rot,k} \\
x_{\rm rot,k+1} \\
\vdots \\ 
x_{\rm rot,n} 
\end{bmatrix}
= x_{\rm rot} 
\end{align}</m>

In our example, this gives us the following plot of <m>\textstyle \tilde{x}</m> (using <m>\textstyle n=2, k=1</m>):

<img src="{{site.baseurl}}/images/PCA-xtilde.png" width="100%">

However, since the final <m>\textstyle n-k</m> components of <m>\textstyle \tilde{x}</m> as defined above would
always be zero, there is no need to keep these zeros around, and so we
define <m>\textstyle \tilde{x}</m> as a <m>\textstyle k</m>-dimensional vector with just the first <m>\textstyle k</m> (non-zero) components. 

This also explains why we wanted to express our data in the <m>\textstyle u_1, u_2, \ldots, u_n</m> basis:
Deciding which components to keep becomes just keeping the top <m>\textstyle k</m> components.  When we
do this, we also say that we are "retaining the top <m>\textstyle k</m> PCA (or principal) components."

### Recovering an Approximation of the Data ###

Now, <m>\textstyle \tilde{x} \in \Re^k</m> is a lower-dimensional, "compressed" representation
of the original <m>\textstyle x \in \Re^n</m>.  Given <m>\textstyle \tilde{x}</m>, how can we recover an approximation <m>\textstyle \hat{x}</m> to 
the original value of <m>\textstyle x</m>?  From an earlier section, we know that <m>\textstyle x = U x_{\rm rot}</m>.  Further, 
we can think of <m>\textstyle \tilde{x}</m> as an approximation to <m>\textstyle x_{\rm rot}</m>, where we have
set the last <m>\textstyle n-k</m> components to zeros.  Thus, given <m>\textstyle \tilde{x} \in \Re^k</m>, we can 
pad it out with <m>\textstyle n-k</m> zeros to get our approximation to <m>\textstyle x_{\rm rot} \in \Re^n</m>.  Finally, we pre-multiply
by <m>\textstyle U</m> to get our approximation to <m>\textstyle x</m>.  Concretely, we get 

<m>\begin{align}
\hat{x}  = U \begin{bmatrix} \tilde{x}_1 \\ \vdots \\ \tilde{x}_k \\ 0 \\ \vdots \\ 0 \end{bmatrix}  
= \sum_{i=1}^k u_i \tilde{x}_i. 
\end{align}</m>

The final equality above comes from the definition of <m>\textstyle U</m> given earlier. 
(In a practical implementation, we wouldn't actually zero pad <m>\textstyle \tilde{x}</m> and then multiply
by <m>\textstyle U</m>, since that would mean multiplying a lot of things by zeros; instead, we'd just 
multiply <m>\textstyle \tilde{x} \in \Re^k</m> with the first <m>\textstyle k</m> columns of <m>\textstyle U</m> as in the final expression above.)
Applying this to our dataset, we get the following plot for <m>\textstyle \hat{x}</m>:

<img src="{{site.baseurl}}/images/PCA-xhat.png" width="100%">

We are thus using a 1 dimensional approximation to the original dataset. 

If you are training an autoencoder or other unsupervised feature learning algorithm,
the running time of your algorithm will depend on the dimension of the input.  If you feed <m>\textstyle \tilde{x} \in \Re^k</m>
into your learning algorithm instead of <m>\textstyle x</m>, then you'll be training on a lower-dimensional
input, and thus your algorithm might run significantly faster.  For many datasets,
the lower dimensional <m>\textstyle \tilde{x}</m> representation can be an extremely good approximation 
to the original, and using PCA this way can significantly speed up your algorithm while
introducing very little approximation error.

### Number of components to retain ###

How do we set <m>\textstyle k</m>; i.e., how many PCA components should we retain?  In our
simple 2 dimensional example, it seemed natural to retain 1 out of the 2
components, but for higher dimensional data, this decision is less trivial.  If <m>\textstyle k</m> is
too large, then we won't be compressing the data much; in the limit of <m>\textstyle k=n</m>,
then we're just using the original data (but rotated into a different basis). Conversely, if <m>\textstyle k</m> is too small, then we might be using a very bad
approximation to the data. 

To decide how to set <m>\textstyle k</m>, we will usually look at the '''percentage of variance retained''' 
for different values of <m>\textstyle k</m>.  Concretely, if <m>\textstyle k=n</m>, then we have
an exact approximation to the data, and we say that 100% of the variance is
retained.  I.e., all of the variation of the original data is retained. Conversely, if <m>\textstyle k=0</m>, then we are approximating all the data with the zero vector,
and thus 0% of the variance is retained. 

More generally, let <m>\textstyle \lambda_1, \lambda_2, \ldots, \lambda_n</m> be the eigenvalues 
of <m>\textstyle \Sigma</m> (sorted in decreasing order), so that <m>\textstyle \lambda_j</m> is the eigenvalue
corresponding to the eigenvector <m>\textstyle u_j</m>.  Then if we retain <m>\textstyle k</m> principal components, 
the percentage of variance retained is given by:

<m>\begin{align}
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^n \lambda_j}.
\end{align}</m>

In our simple 2D example above, <m>\textstyle \lambda_1 = 7.29</m>, and <m>\textstyle \lambda_2 = 0.69</m>.  Thus,
by keeping only <m>\textstyle k=1</m> principal components, we retained <m>\textstyle 7.29/(7.29+0.69) = 0.913</m>,
or 91.3% of the variance.

A more formal definition of percentage of variance retained is beyond the scope
of these notes.  However, it is possible to show that <m>\textstyle \lambda_j =
\sum_{i=1}^m x_{\rm rot,j}^2</m>.  Thus, if <m>\textstyle \lambda_j \approx 0</m>, that shows that <m>\textstyle x_{\rm rot,j}</m> is usually near 0 anyway, and we lose relatively little by
approximating it with a constant 0.  This also explains why we retain the top principal
components (corresponding to the larger values of <m>\textstyle \lambda_j</m>) instead of the bottom
ones.  The top principal components  <m>\textstyle x_{\rm rot,j}</m> are the ones that're more variable and that take on larger values, and for which we would incur a greater approximation error if we were to set them to zero. 

In the case of images, one common heuristic is to choose <m>\textstyle k</m> so as to retain 99% of
the variance.  In other words, we pick the smallest value of <m>\textstyle k</m> that satisfies 

<m>\begin{align}
\frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^n \lambda_j} \geq 0.99. 
\end{align}</m>

Depending on the application, if you are willing to incur some 
additional error, values in the 90-98% range are also sometimes used.  When you
describe to others how you applied PCA, saying that you chose <m>\textstyle k</m> to retain 95% of
the variance will also be a much more easily interpretable description than saying
that you retained 120 (or whatever other number of) components.

### PCA on Images ###
For PCA to work, usually we want each of the features <m>\textstyle x_1, x_2, \ldots, x_n</m>
to have a similar range of values to the others (and to have a mean close to
zero).  If you've used PCA on other applications before, you may therefore have
separately pre-processed each feature to have zero mean and unit variance, by
separately estimating the mean and variance of each feature <m>\textstyle x_j</m>.  However,
this isn't the pre-processing that we will apply to most types of images.  Specifically,
suppose we are training our algorithm on '''natural images''', so that <m>\textstyle x_j</m> is
the value of pixel <m>\textstyle j</m>.  By "natural images," we informally mean the type of image that
a typical animal or person might see over their lifetime.

Note: Usually we use images of outdoor scenes with grass, trees, etc., and cut out small (say 16x16) image patches randomly from these to train the algorithm.  But in practice most feature learning algorithms are extremely robust to the exact type of image  it is trained on, so most images taken with a normal camera, so long as they aren't excessively blurry or have strange artifacts, should work.  

When training on natural images, it makes little sense to estimate a separate mean and
variance for each pixel, because the statistics in one part
of the image should (theoretically) be the same as any other.  
This property of images is called '''stationarity.''' 

In detail, in order for PCA to work well, informally we require that (i) The
features have approximately zero mean, and (ii) The different features have
similar variances to each other.  With natural images, (ii) is already
satisfied even without variance normalization, and so we won't perform any 
variance normalization.  
(If you are training on audio data---say, on
spectrograms---or on text data---say, bag-of-word vectors---we will usually not perform
variance normalization either.)  
In fact, PCA is invariant to the scaling of
the data, and will return the same eigenvectors regardless of the scaling of
the input.  More formally, if you multiply each feature vector <m>\textstyle x</m> by some
positive number (thus scaling every feature in every training example by the
same number), PCA's output eigenvectors will not change.  

So, we won't use variance normalization.  The only normalization we need to
perform then is mean normalization, to ensure that the features have a mean
around zero.  Depending on the application, very often we are not interested
in how bright the overall input image is.  For example, in object recognition
tasks, the overall brightness of the image doesn't affect what objects
there are in the image.  More formally, we are not interested in the
mean intensity value of an image patch; thus, we can subtract out this value,
as a form of mean normalization.  

Concretely, if <m>\textstyle x^{(i)} \in \Re^{n}</m> are the (grayscale) intensity values of
a 16x16 image patch (<m>\textstyle n=256</m>), we might normalize the intensity of each image <m>\textstyle x^{(i)}</m> as follows: 

<m>\mu^{(i)} := \frac{1}{n} \sum_{j=1}^n x^{(i)}_j</m>

<m>x^{(i)}_j := x^{(i)}_j - \mu^{(i)}</m>

for all <m>\textstyle j</m>
   
Note that the two steps above are done separately for each image <m>\textstyle x^{(i)}</m>,
and that <m>\textstyle \mu^{(i)}</m> here is the mean intensity of the image <m>\textstyle x^{(i)}</m>.  In particular,
this is not the same thing as estimating a mean value separately for each pixel <m>\textstyle x_j</m>.

If you are training your algorithm on images other than natural images (for example, images of handwritten characters, or images of single isolated objects centered against a white background), other types of normalization might be worth considering, and the best choice may be application dependent. But when training on natural images, using the per-image mean normalization method as given in the equations above would be a reasonable default.

### Whitening ###

We have used PCA to reduce the dimension of the data.  There is a closely related
preprocessing step called **whitening** (or, in some other literatures, **sphering**)
which is needed for some algorithms.  If we are training on images,
the raw input is redundant, since adjacent pixel values
are highly correlated.  The goal of whitening is to make the input less redundant; more formally,
our desiderata are that our learning algorithms sees a training input where (i) the features are less
correlated with each other, and (ii) the features all have the same variance.

### 2D example ###

We will first describe whitening using our previous 2D example.  We will then 
describe how this can be combined with smoothing, and finally how to combine
this with PCA. 

How can we make our input features uncorrelated with each other?  We had
already done this when computing <m>\textstyle x_{\rm rot}^{(i)} = U^Tx^{(i)}</m>.  
Repeating our previous figure, our plot for <m>\textstyle x_{\rm rot}</m> was:

<img src="{{site.baseurl}}/images/PCA-rotated.png" width="100%">

The covariance matrix of this data is given by:

<m>\begin{align}
\begin{bmatrix}
7.29 &amp; 0  \\
0 &amp; 0.69
\end{bmatrix}.
\end{align}</m>

(Note: Technically, many of the
statements in this section about the "covariance" will be true only if the data
has zero mean.  In the rest of this section, we will take this assumption as
implicit in our statements.  However, even if the data's mean isn't exactly zero, 
the intuitions we're presenting here still hold true, and so this isn't something
that you should worry about.)

It is no accident that the diagonal values are <m>\textstyle \lambda_1</m> and <m>\textstyle \lambda_2</m>. Further, the off-diagonal entries are zero; thus, <m>\textstyle x_{\rm rot,1}</m> and <m>\textstyle x_{\rm rot,2}</m> are uncorrelated, satisfying one of our desiderata for whitened data (that the features be less correlated).

To make each of our input features have unit variance, we can simply rescale
each feature <m>\textstyle x_{\rm rot,i}</m> by <m>\textstyle 1/\sqrt{\lambda_i}</m>.  Concretely, we define
our whitened data <m>\textstyle x_{\rm PCAwhite} \in \Re^n</m> as follows: 

<m>\begin{align}
x_{\rm PCAwhite,i} = \frac{x_{\rm rot,i} }{\sqrt{\lambda_i}}.   
\end{align}</m>

Plotting <m>\textstyle x_{\rm PCAwhite}</m>, we get:

<img src="{{site.baseurl}}/images/PCA-whitened.png" width="100%">

This data now has covariance equal to the identity matrix <m>\textstyle I</m>.  We say that <m>\textstyle x_{\rm PCAwhite}</m> is our **PCA whitened** version of the data: The 
different components of <m>\textstyle x_{\rm PCAwhite}</m> are uncorrelated and have
unit variance. 

**Whitening combined with dimensionality reduction.** 
If you want to have data that is whitened and which is lower dimensional than the original input, you can also optionally keep only the top <m>\textstyle k</m> components of <m>\textstyle x_{\rm PCAwhite}</m>.  When we combine PCA whitening with regularization
(described later), the last few components of <m>\textstyle x_{\rm PCAwhite}</m> will be nearly zero anyway, and thus can safely be dropped.

### ZCA Whitening ### 
Finally, it turns out that this way of getting the 
data to have covariance identity <m>\textstyle I</m> isn't unique. 
Concretely, if <m>\textstyle R</m> is any orthogonal matrix, so that it satisfies <m>\textstyle RR^T = R^TR = I</m> (less formally,
if <m>\textstyle R</m> is a rotation/reflection matrix), then <m>\textstyle R \,x_{\rm PCAwhite}</m> will also have identity covariance. 
 
In **ZCA whitening**,
we choose <m>\textstyle R = U</m>.  We define 

<m>\begin{align}
x_{\rm ZCAwhite} = U x_{\rm PCAwhite}
\end{align}</m>

Plotting <m>\textstyle x_{\rm ZCAwhite}</m>, we get: 

<img src="{{site.baseurl}}/images/ZCA-whitened.png" width="100%">

It can be shown that out of all possible choices for <m>\textstyle R</m>, 
this choice of rotation causes <m>\textstyle x_{\rm ZCAwhite}</m> to be as close as possible to the 
original input data <m>\textstyle x</m>.  

When using ZCA whitening (unlike PCA whitening), we usually keep all <m>\textstyle n</m> dimensions
of the data, and do not try to reduce its dimension.

### Regularizaton ###
When implementing PCA whitening or ZCA whitening in practice, sometimes some
of the eigenvalues <m>\textstyle \lambda_i</m> will be numerically close to 0, and thus the scaling
step where we divide by <m>\sqrt{\lambda_i}</m> would involve dividing by a value close to zero; this 
may cause the data to blow up (take on large values) or otherwise be numerically unstable.  In practice, we 
therefore implement this scaling step using 
a small amount of regularization, and add a small constant <m>\textstyle \epsilon</m> 
to the eigenvalues before taking their square root and inverse:

<m>\begin{align}
x_{\rm PCAwhite,i} = \frac{x_{\rm rot,i} }{\sqrt{\lambda_i + \epsilon}}.
\end{align}</m>

When <m>\textstyle x</m> takes values around <m>\textstyle [-1,1]</m>, a value of <m>\textstyle \epsilon \approx 10^{-5}</m>
might be typical. 

For the case of images, adding <m>\textstyle \epsilon</m> here also has the effect of slightly smoothing (or low-pass
filtering) the input image.  This also has a desirable effect of removing aliasing artifacts
caused by the way pixels are laid out in an image, and can improve the features learned 
(details are beyond the scope of these notes). 

ZCA whitening is a form of pre-processing of the data that maps it
from <m>\textstyle x</m> to <m>\textstyle x_{\rm ZCAwhite}</m>.
It turns out that this is also a rough model of how the biological
eye (the retina) processes images.  Specifically, as your eye
perceives images, most adjacent "pixels" in your eye will perceive
very similar values, since adjacent parts of an image tend to be
highly correlated in intensity.  It is thus wasteful for your eye
to have to transmit every pixel separately (via your optic nerve)
to your brain.  Instead, your retina performs a decorrelation
operation (this is done via retinal neurons that compute a
function called "on center, off surround/off center, on surround")
which is similar to that performed by ZCA.  This results in a less
redundant representation of the input image, which is then
transmitted to your brain.

###Implementing PCA Whitening###

In this section, we summarize the PCA, PCA whitening and ZCA whitening algorithms,
and also describe how you can implement them using efficient linear algebra libraries.

First, we need to ensure that the data has (approximately) zero-mean. For natural images, we achieve this (approximately) by subtracting the mean value of each image patch.

We achieve this by computing the mean for each patch and subtracting it for each patch. In Matlab, we can do this by using

{% highlight matlab %}
 avg = mean(x, 1);     % Compute the mean pixel intensity value separately for each patch. 
 x = x - repmat(avg, size(x, 1), 1);
{% endhighlight %}

Next, we need to compute <m>\textstyle \Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)})(x^{(i)})^T</m>.  If you're implementing this in Matlab (or even if you're implementing this in C++, Java, etc., but have access to an efficient linear algebra library), doing it as an explicit sum is inefficient. Instead, we can compute this in one fell swoop as 

{% highlight matlab %}
 sigma = x * x' / size(x, 2);
{% endhighlight %}

(Check the math yourself for correctness.) 
Here, we assume that <m>x</m> is a data structure that contains one training example per column (so, <m>x</m> is a <m>\textstyle n</m>-by-<m>\textstyle m</m> matrix). 

Next, PCA computes the eigenvectors of <m>\Sigma</m>.  One could do this using the Matlab <tt>eig</tt> function.  However, because <m>\Sigma</m> is a symmetric positive semi-definite matrix, it is more numerically reliable to do this using the <tt>svd</tt> function. Concretely, if you implement 

{% highlight matlab %}
 [U,S,V] = svd(sigma);
{% endhighlight %}

then the matrix <m>U</m> will contain the eigenvectors of <m>\Sigma</m> (one eigenvector per column,  sorted in order from top to bottom eigenvector), and the diagonal entries of the matrix <m>S</m> will contain the corresponding eigenvalues (also sorted in decreasing order).  The matrix <m>V</m> will be equal to transpose of <m>U</m>, and can be safely ignored.

(Note: The <tt>svd</tt> function actually computes the singular vectors and singular values of a matrix, which for the special case of a symmetric positive semi-definite matrix---which is all that we're concerned with here---is equal to its eigenvectors and eigenvalues.  A full discussion of singular vectors vs. eigenvectors is beyond the scope of these notes.)

Finally, you can compute <m>\textstyle x_{\rm rot}</m> and <m>\textstyle \tilde{x}</m> as follows:

{% highlight matlab %}
 xRot = U' * x;          % rotated version of the data. 
 xTilde = U(:,1:k)' * x; % reduced dimension representation of the data, 
                         % where k is the number of eigenvectors to keep
{% endhighlight %}

This gives your PCA representation of the data in terms of <m>\textstyle \tilde{x} \in \Re^k</m>. 
Incidentally, if <m>x</m> is a <m>\textstyle n</m>-by-<m>\textstyle m</m> matrix containing all your training data, this is a vectorized
implementation, and the expressions
above work too for computing <m>x_{\rm rot}</m> and <m>\tilde{x}</m> for your entire training set
all in one go.  The resulting <m>x_{\rm rot}</m> and <m>\tilde{x}</m> will have one column corresponding to each training example. 

To compute the PCA whitened data <m>\textstyle x_{\rm PCAwhite}</m>, use 

{% highlight matlab %}
 xPCAwhite = diag(1./sqrt(diag(S) + epsilon)) * U' * x;
{% endhighlight %}

Since <m>S</m>'s diagonal contains the eigenvalues <m>\textstyle \lambda_i</m>, 
this turns out to be a compact way 
of computing <m>\textstyle x_{\rm PCAwhite,i} = \frac{x_{\rm rot,i} }{\sqrt{\lambda_i}}</m>
simultaneously for all <m>\textstyle i</m>.  

Finally, you can also compute the ZCA whitened data <m>\textstyle x_{\rm ZCAwhite}</m> as:

{% highlight matlab %}
 xZCAwhite = U * diag(1./sqrt(diag(S) + epsilon)) * U' * x;
{% endhighlight %}


