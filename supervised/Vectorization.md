---
layout: post
title:  "Vectorization"
date:   2013-08-28 13:32:12
categories: supervised
---

For small jobs like the housing prices data we used for linear regression, your code does not need to be extremely fast.  However, if your implementation for Exercise 1A or 1B used a for-loop as suggested, it is probably too slow to work well for large problems that are more interesting.  This is because looping over the examples (or any other elements) sequentially in MATLAB is slow.  To avoid for-loops, we want to rewrite our code to make use of optimized vector and matrix operations so that MATLAB will execute it quickly.  (This is also useful for other languages, including Python and C/C++ --- we want to re-use optimized operations when possible.)

Following are some examples for how to vectorize various operations in MATLAB.

###Example: Many matrix-vector products###

Frequently we want to compute matrix-vector products for many vectors at once, such as when we compute <m>\theta^\top x^{(i)}</m> for each example in a dataset (where <m>\theta</m> may be a 2D matrix, or a vector itself).  We can form a matrix <m>X</m> containing our entire dataset  by concatenating the examples <m>x^{(i)}</m> to form the columns of <m>X</m>:

<m>
X = \left[\begin{array}{cccc}
  | &amp; |  &amp;  | &amp; | \\
  x^{(1)} &amp; x^{(2)} &amp; \cdots &amp; x^{(m)}\\
    | &amp; |  &amp;  | &amp; |\end{array}\right]
</m>

With this notation, we can compute <m>y^{(i)} = W x^{(i)}</m> for all <m>x^{(i)}</m> at once as:

<m>
\left[\begin{array}{cccc}
| &amp; |  &amp;  | &amp; | \\
y^{(1)} &amp; y^{(2)} &amp; \cdots &amp; y^{(m)}\\
| &amp; |  &amp;  | &amp; |\end{array}\right] = Y = W X
</m>

So, when performing linear regression, we can use <m>\theta^\top X</m> to avoid looping over all of our examples to compute <m>y^{(i)} = \theta^\top X^{(i)}</m>.

###Example:  normalizing many vectors###

Suppose we have many vectors <m>x^{(i)}</m> concatenated into a matrix <m>X</m> as above, and we want to compute <m>y^{(i)} = x^{(i)}/||x^{(i)}||_2</m> for all of the <m>x^{(i)}</m>.  This may be done using several of MATLAB's array operations:

{% highlight matlab %}
  X_norm = sqrt( sum(X.^2,1) );
  Y = bsxfun(@rdivide, X, X_norm);
{% endhighlight %}

This code squares all of the elements of X, then sums along the first dimension (the rows) of the result, and finally takes the square root of each element.  This leaves us with a 1-by-m matrix containing <m>||x^{(i)}||_2</m>.  The `bsxfun` routine can be thought of as expanding or cloning <m>{X\text{norm}}</m> so that it has the same dimension as <m>X</m> before applying an element-wise binary function.  In the example above it divides every element <m>X_{ji} = x_j^{(i)}</m> by the corresponding column in <m>X\text{norm}</m>, leaving us with <m>Y_{ji} = X_{ji} / {X\text{norm}}_i = x_j^{(i)}/||x^{(i)}||_2</m> as desired.  `bsxfun` can be used with almost any binary element-wise function (e.g., @plus, @ge, or @eq).  See the `bsxfun` docs!

###Example:  matrix multiplication in gradient computations###

In our linear regression gradient computation, we have a summation of the form:

<m>
\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_i x_j^{(i)} (\hat{y}^{(i)} - y^{(i)}). 
</m>

Whenever we have a summation over a single index (in this case <m>i</m>) with several other fixed indices (in this case <m>j</m>) we can often rephrase the computation as a matrix multiply since <m>[A B]_{jk} = \sum_i A_{ji} B_{ik}</m>.  If <m>y</m> and <m>\hat{y}</m> are column vectors (so <m>y_i \equiv y^{(i)}</m>), then with this template we can rewrite the above summation as:

<m>
\frac{\partial J(\theta; X,y)}{\partial \theta_j} = \sum_i X_{ji} (\hat{y}_i - y_i) = [X (\hat{y} - y)]_j.
</m>

Thus, to perform the entire computation for every <m>j</m> we can just compute <m>X (\hat{y} - y)</m>.  In MATLAB:
{% highlight matlab %}
% X(j,i) = j'th coordinate of i'th example.
% y(i) = i'th value to be predicted;  y is a column vector.
% theta = vector of parameters

y_hat = theta'*X; % so y_hat(i) = theta' * X(:,i).  Note that y_hat is a *row-vector*.
g = X*(y_hat' - y);
{% endhighlight %}

### Exercise 1A and 1B Redux###

Go back to your Exercise 1A and 1B code.  In the `ex1a_linreg.m` file and `ex1b_logreg.m` file you will find commented-out code that calls `minFunc` using `linear_regression_vec.m` and `logistic_regression_vec.m` (respectively) instead of `linear_regression.m` and `logistic_regression.m`.  For this exercise, fill in the `linear_regression_vec.m` and `logistic_regression_vec.m` files with a vectorized implementation of your previous solutions.  Uncomment the calling code in `ex1a_linreg.m` and `ex1b_logreg.m` and compare the running times of each implementation.  Verify that you get similar results to your original solutions!

