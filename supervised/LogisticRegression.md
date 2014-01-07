---
layout: post
title:  "Logistic Regression"
date:   2013-08-28 13:32:12
categories: supervised
---
Previously we learned how to predict continuous-valued quantities (e.g., housing prices) as a linear function of input values (e.g., the size of the house).  Sometimes we will instead wish to predict a discrete variable such as predicting whether a grid of pixel intensities represents a "0" digit or a "1" digit.  This is a classification problem.  Logistic regression is a simple classification algorithm for learning to make such decisions.  

In linear regression we tried to predict the value of <m>y^{(i)}</m> for the
<m>i</m>'th example <m>x^{(i)}</m> using a linear function <m>y = h_\theta(x) =
\theta^\top x.</m>.  This is clearly not a great solution for predicting
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

