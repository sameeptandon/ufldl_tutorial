---
layout: post
title:  "Softmax Regression"
date:   2013-08-28 13:32:12
categories: supervised
---
### Introduction ###

Softmax regression (or multinomial logistic regression) is a generalization of logistic regression to the case where we want to handle multiple classes.  In logistic regression we assumed that the labels were binary:  <m>y^{(i)} \in \{0,1\}</m>.  We used such a classifier to distinguish between two kinds of hand-written digits.  Softmax regression allows us to handle <m>y^{(i)} \in \{1,\ldots,K\}</m> where <m>K</m> is the number of classes.

Recall that in logistic regression, we had a training set <m>\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}</m> of <m>m</m> labeled examples, where the input features are <m>x^{(i)} \in \Re^{n}</m>. With logistic regression, we were in the binary classification setting, so the labels 
were <m>y^{(i)} \in \{0,1\}</m>.  Our hypothesis took the form:

<m>\begin{align}
h_\theta(x) = \frac{1}{1+\exp(-\theta^\top x)},
\end{align}</m>

and the model parameters <m>\theta</m> were trained to minimize
the cost function

<m>
\begin{align}
J(\theta) = -\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) \right]
\end{align}
</m>

In the softmax regression setting, we are interested in multi-class
classification (as opposed to only binary classification), and so the label <m>y</m> can take on <m>K</m> different values, rather than only
two.  Thus, in our training set <m>\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}</m>, we now have that <m>y^{(i)} \in \{1, 2, \ldots, K\}</m>.  (Note that our convention will be to index the classes starting from 1, rather than from 0.)  For example, in the MNIST digit recognition task, we would have <m>K=10</m> different classes.

Given a test input <m>x</m>, we want our hypothesis to estimate
the probability that <m>P(y=k | x)</m> for each value of <m>k = 1, \ldots, K</m>.
I.e., we want to estimate the probability of the class label taking
on each of the <m>K</m> different possible values.  Thus, our hypothesis
will output a <m>K</m>-dimensional vector (whose elements sum to 1) giving
us our <m>K</m> estimated probabilities.  Concretely, our hypothesis <m>h_{\theta}(x)</m> takes the form:

<m>
\begin{align}
h_\theta(x) =
\begin{bmatrix}
P(y = 1 | x; \theta) \\
P(y = 2 | x; \theta) \\
\vdots \\
P(y = K | x; \theta)
\end{bmatrix}
=
\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) }}
\begin{bmatrix}
\exp(\theta^{(1)\top} x ) \\
\exp(\theta^{(2)\top} x ) \\
\vdots \\
\exp(\theta^{(K)\top} x ) \\
\end{bmatrix}
\end{align}
</m>

Here <m>\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)} \in \Re^{n}</m> are the
parameters of our model. Notice that the term <m>\frac{1}{ \sum_{j=1}^{K}{\exp(\theta^{(j)\top} x) } } </m>
normalizes the distribution, so that it sums to one. 

For convenience, we will also write <m>\theta</m> to denote all the
parameters of our model.  When you implement softmax regression, it is usually
convenient to represent <m>\theta</m> as a <m>n</m>-by-<m>K</m> matrix obtained by concatenating <m>\theta^{(1)}, \theta^{(2)}, \ldots, \theta^{(K)}</m> into columns, so that

<m>
\theta = \left[\begin{array}{cccc}| &amp; | &amp; | &amp; | \\
\theta^{(1)} &amp; \theta^{(2)} &amp; \cdots &amp; \theta^{(K)} \\
| &amp; | &amp; | &amp; |
\end{array}\right].
</m>

### Cost Function ###

We now describe the cost function that we'll use for softmax regression.  In the equation below, <m>1\{\cdot\}</m> is
the '''indicator function,''' so that <m>1\{\hbox{a true statement}\}=1</m>, and <m>1\{\hbox{a false statement}\}=0</m>.
For example, <m>1\{2+2=4\}</m> evaluates to 1; whereas <m>1\{1+1=5\}</m> evaluates to 0. Our cost function will be:

<m>
\begin{align}
J(\theta) = - \left[ \sum_{i=1}^{m} \sum_{k=1}^{K}  1\left\{y^{(i)} = k\right\} \log \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}\right]
\end{align}
</m>

Notice that this generalizes the logistic regression cost function, which could also have been written:

<m>
\begin{align}
J(\theta) &amp;= - \left[ \sum_{i=1}^m   (1-y^{(i)}) \log (1-h_\theta(x^{(i)})) + y^{(i)} \log h_\theta(x^{(i)}) \right] \\
&amp;= - \left[ \sum_{i=1}^{m} \sum_{k=0}^{1} 1\left\{y^{(i)} = k\right\} \log P(y^{(i)} = k | x^{(i)} ; \theta) \right]
\end{align}
</m>

The softmax cost function is similar, except that we now sum over the <m>K</m> different possible values
of the class label.  Note also that in softmax regression, we have that

<m> P(y^{(i)} = k | x^{(i)} ; \theta) = \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) }
</m>.
We cannot solve for the minimum of <m>J(\theta)</m> analytically, and thus as usual we'll resort to an iterative
optimization algorithm.  Taking derivatives, one can show that the gradient is:

<m>
\begin{align}
\nabla_{\theta^{(k)}} J(\theta) = - \sum_{i=1}^{m}{ \left[ x^{(i)} \left( 1\{ y^{(i)} = k\}  - P(y^{(i)} = k | x^{(i)}; \theta) \right) \right]  }
\end{align}
</m>


Recall the meaning of the "<m>\nabla_{\theta^{(k)}}</m>" notation.  In particular, <m>\nabla_{\theta^{(k)}} J(\theta)</m>
is itself a vector, so that its <m>j</m>-th element is <m>\frac{\partial J(\theta)}{\partial \theta_{lk}}</m>
the partial derivative of <m>J(\theta)</m> with respect to the <m>j</m>-th element of <m>\theta^{(k)}</m>. 

Armed with this formula for the derivative, one can then plug it into a standard optimization package and have it
minimize <m>J(\theta)</m>. 

<!--
When implementing softmax regression, we will typically use a modified version of the cost function described above;
specifically, one that incorporates weight decay.  We describe the motivation and details below. -->

### Properties of softmax regression parameterization ###

Softmax regression has an unusual property that it has a "redundant" set of parameters.  To explain what this means, 
suppose we take each of our parameter vectors <m>\theta^{(j)}</m>, and subtract some fixed vector <m>\psi</m>
from it, so that every <m>\theta^{(j)}</m> is now replaced with <m>\theta^{(j)} - \psi</m> 
(for every <m>j=1, \ldots, k</m>).  Our hypothesis
now estimates the class label probabilities as

<m>
\begin{align}
P(y^{(i)} = k | x^{(i)} ; \theta)
&amp;= \frac{\exp((\theta^{(k)}-\psi)^\top x^{(i)})}{\sum_{j=1}^K \exp( (\theta^{(j)}-\psi)^\top x^{(i)})}  \\
&amp;= \frac{\exp(\theta^{(k)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)}) \exp(-\psi^\top x^{(i)})} \\
&amp;= \frac{\exp(\theta^{(k)\top} x^{(i)})}{\sum_{j=1}^K \exp(\theta^{(j)\top} x^{(i)})}.
\end{align}
</m>

In other words, subtracting <m>\psi</m> from every <m>\theta^{(j)}</m>
does not affect our hypothesis' predictions at all!  This shows that softmax
regression's parameters are "redundant."  More formally, we say that our
softmax model is '''overparameterized,''' meaning that for any hypothesis we might
fit to the data, there are multiple parameter settings that give rise to exactly
the same hypothesis function <m>h_\theta</m> mapping from inputs <m>x</m>
to the predictions. 

Further, if the cost function <m>J(\theta)</m> is minimized by some
setting of the parameters <m>(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(k)})</m>,
then it is also minimized by <m>(\theta^{(1)} - \psi, \theta^{(2)} - \psi,\ldots,
\theta^{(k)} - \psi)</m> for any value of <m>\psi</m>.  Thus, the
minimizer of <m>J(\theta)</m> is not unique.  (Interestingly,  <m>J(\theta)</m> is still convex, and thus gradient descent will not run into local optima problems.  But the Hessian is singular/non-invertible, which causes a straightforward implementation of Newton's method to run into numerical problems.) 

Notice also that by setting <m>\psi = \theta^{(K)}</m>, one can always
replace <m>\theta^{(K)}</m> with <m>\theta^{(K)} - \psi = \vec{0}</m> (the vector of all
0's), without affecting the hypothesis.  Thus, one could "eliminate" the vector
of parameters <m>\theta^{(K)}</m> (or any other <m>\theta^{(k)}</m>, for
any single value of <m>k</m>), without harming the representational power
of our hypothesis.  Indeed, rather than optimizing over the <m>K\cdot n</m>
parameters <m>(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(K)})</m> (where <m>\theta^{(k)} \in \Re^{n}</m>), one can instead set <m>\theta^{(K)} = \vec{0}</m> and optimize only with respect to the <m>K \cdot n</m> remaining parameters. 

<!--
In practice, however, it is often cleaner and simpler to implement the version which keeps
all the parameters <m>(\theta^{(1)}, \theta^{(2)},\ldots, \theta^{(n)})</m>, without
arbitrarily setting one of them to zero.  But we will
make one change to the cost function: Adding weight decay.  This will take care of
the numerical problems associated with softmax regression's overparameterized representation.

### Weight Decay ###

We will modify the cost function by adding a weight decay term 
<m>\textstyle \frac{\lambda}{2} \sum_{i=1}^k \sum_{j=0}^{n} \theta_{ij}^2</m>
which penalizes large values of the parameters.  Our cost function is now

<m>
\begin{align}
J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} \sum_{j=1}^{k} 1\left\{y^{(i)} = j\right\} \log \frac{e^{\theta^{(j)\top} x^{(i)}}}{\sum_{l=1}^k e^{ \theta^{(l)\top} x^{(i)} }}  \right]
              + \frac{\lambda}{2} \sum_{i=1}^k \sum_{j=0}^n \theta_{ij}^2
\end{align}
</m>

With this weight decay term (for any <m>\lambda > 0</m>), the cost function
<m>J(\theta)</m> is now strictly convex, and is guaranteed to have a
unique solution.  The Hessian is now invertible, and because <m>J(\theta)</m> is 
convex, algorithms such as gradient descent, L-BFGS, etc. are guaranteed
to converge to the global minimum.

To apply an optimization algorithm, we also need the derivative of this
new definition of <m>J(\theta)</m>.  One can show that the derivative is:
<m>
\begin{align}
\nabla_{\theta^{(j)}} J(\theta) = - \frac{1}{m} \sum_{i=1}^{m}{ \left[ x^{(i)} ( 1\{ y^{(i)} = j\}  - P(y^{(i)} = j | x^{(i)}; \theta) ) \right]  } + \lambda \theta^{(j)}
\end{align}
</m>

By minimizing <m>J(\theta)</m> with respect to <m>\theta</m>, we will have a working implementation of softmax regression.
-->

### Relationship to Logistic Regression ###

In the special case where <m>K = 2</m>, one can show that softmax regression reduces to logistic regression.
This shows that softmax regression is a generalization of logistic regression.  Concretely, when <m>K=2</m>,
the softmax regression hypothesis outputs

<m>
\begin{align}
h_\theta(x) &amp;=

\frac{1}{ \exp(\theta^{(1)\top}x)  + \exp( \theta^{(2)\top} x^{(i)} ) }
\begin{bmatrix}
\exp( \theta^{(1)\top} x ) \\
\exp( \theta^{(2)\top} x )
\end{bmatrix}
\end{align}
</m>

Taking advantage of the fact that this hypothesis
is overparameterized and setting <m>\psi = \theta^{(2)}</m>,
we can subtract <m>\theta^{(2)}</m> from each of the two parameters, giving us

<m>
\begin{align}
h(x) &amp;=

\frac{1}{ \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) + \exp(\vec{0}^\top x) }
\begin{bmatrix}
\exp( (\theta^{(1)}-\theta^{(2)})^\top x )
\exp( \vec{0}^\top x ) \\
\end{bmatrix} \\

&amp;=
\begin{bmatrix}
\frac{1}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\frac{\exp( (\theta^{(1)}-\theta^{(2)})^\top x )}{ 1 + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) }
\end{bmatrix} \\

&amp;=
\begin{bmatrix}
\frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
1 - \frac{1}{ 1  + \exp( (\theta^{(1)}-\theta^{(2)})^\top x^{(i)} ) } \\
\end{bmatrix}
\end{align}
</m>


Thus, replacing <m>\theta^{(2)}-\theta^{(1)}</m> with a single parameter vector <m>\theta'</m>, we find
that softmax regression predicts the probability of one of the classes as <m>\frac{1}{ 1  + \exp(- (\theta')^\top x^{(i)} ) }</m>,
and that of the other class as <m>1 - \frac{1}{ 1 + \exp(- (\theta')^\top x^{(i)} ) }</m>, same as logistic regression.

<!--
### Softmax Regression vs. k Binary Classifiers ###

Suppose you are working on a music classification application, and there are
<m>k</m> types of music that you are trying to recognize.  Should you use a
softmax classifier, or should you build <m>k</m> separate binary classifiers using
logistic regression?

This will depend on whether the four classes are ''mutually exclusive.''  For example,
if your four classes are classical, country, rock, and jazz, then assuming each
of your training examples is labeled with exactly one of these four class labels,
you should build a softmax classifier with <m>k=4</m>.
(If there're also some examples that are none of the above four classes,
then you can set <m>k=5</m> in softmax regression, and also have a fifth, "none of the above," class.)

If however your categories are has_vocals, dance, soundtrack, pop, then the
classes are not mutually exclusive; for example, there can be a piece of pop
music that comes from a soundtrack and in addition has vocals.  In this case, it
would be more appropriate to build 4 binary logistic regression classifiers. 
This way, for each new musical piece, your algorithm can separately decide whether
it falls into each of the four categories.

Now, consider a computer vision example, where you're trying to classify images into
three different classes.  (i) Suppose that your classes are indoor_scene,
outdoor_urban_scene, and outdoor_wilderness_scene.  Would you use sofmax regression
or three logistic regression classifiers?  (ii) Now suppose your classes are
indoor_scene, black_and_white_image, and image_has_people.  Would you use softmax
regression or multiple logistic regression classifiers?

In the first case, the classes are mutually exclusive, so a softmax regression
classifier would be appropriate.  In the second case, it would be more appropriate to build
three separate logistic regression classifiers.
-->

### Exercise 1C ###
Starter code for this exercise is included in the [Starter code GitHub Repo](https://github.com/amaas/stanford_dl_ex) in the `ex1/` directory.

In this exercise you will train a classifier to handle all 10 digits in the MNIST dataset.  The code is very similar to that used for Exercise 1B except that it will load the entire MNIST train and test sets (instead of just the 0 and 1 digits), and the labels <m>y^{(i)}</m> have 1 added to them so that <m>y^{(i)} \in \{1,\ldots,10\}</m>.  (The change in the labels allows you to use <m>y^{(i)}</m> as an index into a matrix.)

The code performs the same operations as in Exercise 1B:  it loads the train and test data, adding an intercept term, then calls `minFunc` with the `softmax_regression_vec.m` file as the objective function.  When training is complete, it will print out training and testing accuracies for the 10-class digit recognition problem.

Your task is to implement the `softmax_regression_vec.m` file to compute the softmax objective function <m>J(\theta; X,y)</m> and store it in the variable <m>f</m>.  You must also compute the gradient <m>\nabla_\theta J(\theta; X,y)</m> and store it in the variable <m>g</m>.  Don't forget that `minFunc` supplies the parameters <m>\theta</m> as a vector.  The starter code will reshape <m>\theta</m> into a n-by-(K-1) matrix (for K=10 classes).  You also need to remember to reshape the returned gradient <m>g</m> back into a vector using <m>g=g(:);</m>

You can start out with a for-loop version of the code if necessary to get the gradient right.  (Be sure to use the gradient check debugging strategy covered earlier!)  However, you might find that this implementation is too slow to run the optimizer all the way through.  After you get the gradient right with a slow version of the code, try to vectorize your code as well as possible before running the full experiment.


Here are a few MATLAB tips that you might find useful for implementing or speeding up your code (though these may or may not be useful depending on your implementation strategy):

1. Suppose we have a matrix <m>A</m> and we want to extract a single element from each row, where the column of the element to be extracted from row <m>i</m> is stored in <m>y(i)</m>, where <m>y</m> is a row vector.  We can use the `sub2ind()` function like this:

		I=sub2ind(size(A), 1:size(A,1), y);
		values = A(I);

	This code will take each pair of indices <m>(i,j)</m> where <m>i</m> comes from the second argument and <m>j</m> comes from the corresponding element of the third argument, and compute the corresponding 1D index into <m>A</m> for the <m>(i,j)</m>'th element.  So, <m>I(1)</m> will be the index for the element at location <m>(1,y(1))</m>, and <m>I(2)</m> will be the index for the element at <m>(2,y(2))</m>.

2. When you compute the predicted label probabilities <m>\hat{y}^{(i)}_k = \exp(\theta_{:,k}^\top x^{(i)}) / (\sum^K_{j=1} \exp(\theta_{:,j}^\top x^{(i)}))</m>, try to use matrix multiplications and `bsxfun` to speed up the computation.  For example, once <m>\theta</m> is in matrix form, you can compute the products for every example and the first 9 classes using <m>a = \theta^\top X</m>.  (Recall that the 10th class is left out of <m>\theta</m>, so that <m>a(10,:)</m> is just assumed to be 0.)



