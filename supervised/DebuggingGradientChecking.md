---
layout: post
title:  "Debugging: Gradient Checking"
date:   2013-08-28 13:32:12
categories: supervised
---

So far we have worked with relatively simple algorithms where it is straight-forward to compute the objective function and its gradient with pen-and-paper, and then implement the necessary computations in MATLAB.  For more complex models that we will see later (like the back-propagation method for neural networks), the gradient computation can be notoriously difficult to debug and get right.  Sometimes a subtly buggy implementation will
manage to learn something that can look surprisingly reasonable (while performing less well than a correct implementation).  Thus, even with a
buggy implementation, it may not at all be apparent that anything is amiss.
In this section, we describe a method for numerically checking the derivatives computed
by your code to make sure that your implementation is correct.  Carrying out the
derivative checking procedure described here will significantly increase
your confidence in the correctness of your code.

Suppose we want to minimize <m>\textstyle J(\theta)</m> as a function of <m>\textstyle \theta</m>.
For this example, suppose <m>\textstyle J : \Re \mapsto \Re</m>, so that <m>\textstyle \theta \in \Re</m>.  If we are using `minFunc` or some other optimization algorithm, then we usually have implemented some function <m>\textstyle g(\theta)</m> that purportedly
computes <m>\textstyle \frac{d}{d\theta}J(\theta)</m>.

How can we check if our implementation of <m>\textstyle g</m> is correct?

Recall the mathematical definition of the derivative as:

<m>\begin{align}
\frac{d}{d\theta}J(\theta) = \lim_{\epsilon \rightarrow 0}
\frac{J(\theta+ \epsilon) - J(\theta-\epsilon)}{2 \epsilon}.
\end{align}</m>

Thus, at any specific value of <m>\textstyle \theta</m>, we can numerically approximate the derivative
as follows:

<m>\begin{align}
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}
\end{align}</m>

In practice, we set <m>{\rm EPSILON}</m> to a small constant, say around <m>\textstyle 10^{-4}</m>.
(There's a large range of values of <m>{\rm EPSILON}</m> that should work well, but
we don't set <m>{\rm EPSILON}</m> to be "extremely" small, say <m>\textstyle 10^{-20}</m>,
as that would lead to numerical roundoff errors.)

Thus, given a function <m>\textstyle g(\theta)</m> that is supposedly computing <m>\textstyle \frac{d}{d\theta}J(\theta)</m>, we can now numerically verify its correctness by checking that 

<m>\begin{align}
g(\theta) \approx
\frac{J(\theta+{\rm EPSILON}) - J(\theta-{\rm EPSILON})}{2 \times {\rm EPSILON}}.
\end{align}</m>


The degree to which these two values should approximate each other
will depend on the details of <m>\textstyle J</m>.  But assuming <m>\textstyle {\rm EPSILON} = 10^{-4}</m>,
you'll usually find that the left- and right-hand sides of the above will agree
to at least 4 significant digits (and often many more).

Now, consider the case where <m>\textstyle \theta \in \Re^n</m> is a vector rather than a single real
number (so that we have <m>\textstyle n</m> parameters that we want to learn), and <m>\textstyle J: \Re^n \mapsto \Re</m>. We now generalize our derivative checking procedure to the case where <m>\textstyle \theta</m> may be a vector (as in our linear regression and logistic regression examples).  If ever we are optimizing over several variables or over matrices, we can always pack these parameters into a long vector and use the same method here to check our derivatives.  (This will often need to be done anyway if you want to use off-the-shelf optimization packages.)

Suppose we have a function <m>\textstyle g_i(\theta)</m> that purportedly computes <m>\textstyle \frac{\partial}{\partial \theta_i} J(\theta)</m>; we'd like to check if <m>\textstyle g_i</m>
is outputting correct derivative values.  Let <m>\textstyle \theta^{(i+)} = \theta + {\rm EPSILON} \times \vec{e}_i</m>, where

<m>\begin{align}
\vec{e}_i = \begin{bmatrix}0 \\ 0 \\ \vdots \\ 1 \\ \vdots \\ 0\end{bmatrix}
\end{align}</m>
is the <m>\textstyle i</m>-th basis vector (a
vector of the same dimension as <m>\textstyle \theta</m>, with a "1" in the <m>\textstyle i</m>-th position
and "0"s everywhere else).  So, <m>\textstyle \theta^{(i+)}</m> is the same as <m>\textstyle \theta</m>, except its <m>\textstyle i</m>-th element has been incremented by <m>{\rm EPSILON}</m>.  Similarly, let <m>\textstyle \theta^{(i-)} = \theta - {\rm EPSILON} \times \vec{e}_i</m> be the
corresponding vector with the <m>\textstyle i</m>-th element decreased by <m>{\rm EPSILON}</m>.

We can now numerically verify <m>\textstyle g_i(\theta)</m>'s correctness by checking, for each <m>\textstyle i</m>,
that:

<m>\begin{align}
g_i(\theta) \approx
\frac{J(\theta^{(i+)}) - J(\theta^{(i-)})}{2 \times {\rm EPSILON}}.
\end{align}</m>

### Gradient checker code ###

As an exercise, try implementing the above method to check the gradient of your linear regression and logistic regression functions.  Alternatively, you can use the provided `ex1/grad_check.m` file (which takes arguments similar to `minFunc`) and will check <m>\frac{\partial J(\theta)}{\partial \theta_i}</m> for many random choices of <m>i</m>.

