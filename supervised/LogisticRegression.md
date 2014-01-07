---
layout: post
title:  "Logistic Regression"
date:   2013-08-28 13:32:12
categories: supervised
---
Previously we learned how to predict continuous-valued quantities (e.g., housing prices) as a linear function of input values (e.g., the size of the house).  Sometimes we will instead wish to predict a discrete variable such as predicting whether a grid of pixel intensities represents a "0" digit or a "1" digit.  This is a classification problem.  Logistic regression is a simple classification algorithm for learning to make such decisions.  

In linear regression we tried to predict the value of <m>y^{(i)}</m> for the <m>i</m>'th example <m>x^{(i)}</m> using a linear function <m>y = h_\theta(x) = \theta^\top x.</m>.  This is clearly not a great solution for predicting
binary-valued labels <m>\left(y^{(i)} \in \{0,1\}\right)</m>.  In logistic regression we
use a different hypothesis class to try to predict the probability that a given
example belongs to the "1" class versus the probability that it belongs to the
"0" class.  Specifically, we will try to learn a function of the form:

<m>
\begin{align}
P(y=1|x) &amp;= h_\theta(x) = \frac{1}{1 + \exp(-\theta^\top x)} \equiv \sigma(\theta^\top x),\\
P(y=0|x) &amp;= 1 - P(y=1|x) = 1 - h_\theta(x).
\end{align}
</m>

The function <m>\sigma(z) \equiv \frac{1}{1 + \exp(-z)}</m> is often called the "sigmoid" or "logistic" function -- it is an S-shaped function that "squashes" the value of <m>\theta^\top x</m> into the range <m>[0, 1]</m> so that we may interpret <m>h_\theta(x)</m> as a probability.  Our goal is to search for a value of <m>\theta</m> so that the probability <m>P(y=1|x) = h_\theta(x)</m> is large when <m>x</m> belongs to the "1" class and small when <m>x</m> belongs to the "0" class (so that <m>P(y=0|x)</m> is large).  For a set of training examples with binary labels <m>\{ (x^{(i)}, y^{(i)}) : i=1,\ldots,m\}</m> the following cost function measures how well a given <m>h_\theta</m> does this:

<m>
J(\theta) = - \sum_i \left(y^{(i)} \log( h_\theta(x^{(i)}) ) + (1 - y^{(i)}) \log( 1 - h_\theta(x^{(i)}) ) \right).
</m>

Note that only one of the two terms in the summation is non-zero for each training example (depending on whether the label <m>y^{(i)}</m> is 0 or 1).  When <m>y^{(i)} = 1</m> minimizing the cost function means we need to make <m>h_\theta(x^{(i)})</m> large, and when <m>y^{(i)} = 0</m> we want to make <m>1 - h_\theta</m> large as explained above.  For a full explanation of logistic regression and how this cost function is derived, see the [CS229 Notes](http://cs229.stanford.edu/notes/cs229-notes1.pdf) on supervised learning.

We now have a cost function that measures how well a given hypothesis <m>h_\theta</m> fits our training data.  We can learn to classify our training data by minimizing <m>J(\theta)</m> to find the best choice of <m>\theta</m>.  Once we have done so, we can classify a new test point as "1" or "0" by checking which of these two class labels is most probable: if <m>P(y=1|x) > P(y=0|x)</m> then we label the example as a "1", and "0" otherwise.  This is the same as checking whether <m>h_\theta(x) > 0.5</m>.

To minimize <m>J(\theta)</m> we can use the same tools as for linear regression.  We need to provide a function that computes <m>J(\theta)</m> and <m>\nabla_\theta J(\theta)</m> for any requested choice of <m>\theta</m>.  The derivative of <m>J(\theta)</m> as given above with respect to <m>\theta_j</m> is:

<m>
\frac{\partial J(\theta)}{\partial \theta_j} = \sum_i x^{(i)}_j (h_\theta(x^{(i)}) - y^{(i)}).
</m>

Written in its vector form, the entire gradient can be expressed as:

<m>
\nabla_\theta J(\theta) = \sum_i x^{(i)} (h_\theta(x^{(i)}) - y^{(i)}) 
</m>

This is essentially the same as the gradient for linear regression except that now <m>h_\theta(x) = \sigma(\theta^\top x)</m>.

### Exercise 1B ###
Starter code for this exercise is included in the [Starter Code GitHub Repo](https://github.com/amaas/stanford_dl_ex) in the ex1/ directory.

In this exercise you will implement the objective function and gradient computations for logistic regression and use your code to learn to classify images of digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) as either "0" or "1".  Some examples of these digits are shown below:

<center><img src="{{site.baseurl}}/images/Mnist_01.png"></center>

Each of the digits is is represented by a 28x28 grid of pixel intensities, which we will reformat as a vector <m>x^{(i)}</m> with 28\*28 = 784 elements.  The label is binary, so <m>y^{(i)} \in \{0,1\}</m>.

You will find starter code for this exercise in the `ex1/ex1b_logreg.m` file.  The starter code file performs the following tasks for you:

1.  Calls `ex1_load_mnist.m` to load the MNIST training and testing data.  In addition to loading the pixel values into a matrix <m>X</m> (so that that j'th pixel of the i'th example is <m>X_{ji} = x^{(i)}_j</m>) and the labels into a row-vector <m>y</m>, it will also perform some simple normalizations of the pixel intensities so that they tend to have zero mean and unit variance.  Even though the MNIST dataset contains 10 different digits (0-9), in this exercise we will only load the 0 and 1 digits --- the ex1_load_mnist function will do this for you.

2. The code will append a row of 1's so that <m>\theta_0</m> will act as an intercept term.

3. The code calls `minFunc` with the `logistic_regression.m` file as objective function.  Your job will be to fill in `logistic_regression.m` to return the objective function value and its gradient.

4. After `minFunc` completes, the classification accuracy on the training set and test set will be printed out.

As for the linear regression exercise, you will need to implement `logistic_regression.m` to loop over all of the training examples <m>x^{(i)}</m> and compute the objective <m>J(\theta; X,y)</m>.  Store the resulting objective value into the variable <m>f</m>.  You must also compute the gradient <m>\nabla_\theta J(\theta; X,y)</m> and store it into the variable <m>g</m>.  Once you have completed these tasks, you will be able to run the `ex1b_logreg.m` script to train the classifier and test it.

If your code is functioning correctly, you should find that your classifier is able to achieve 100% accuracy on both the training and testing sets!  It turns out that this is a relatively easy classification problem because 0 and 1 digits tend to look very different.  In future exercises it will be much more difficult to get perfect results like this.

