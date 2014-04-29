---
layout: post
title:  "ICA"
date:   2013-08-28 13:32:12
categories: unsupervised
---
### Introduction ###

If you recall, in [Sparse Coding]({{site.baseurl}}/unsupervised/SparseCoding), we wanted to learn an **over-complete** basis for the data. In particular, this implies that the basis vectors that we learn in sparse coding will not be linearly independent. While this may be desirable in certain situations, sometimes we want to learn a linearly independent basis for the data. In independent component analysis (ICA), this is exactly what we want to do. Further, in ICA, we want to learn not just any linearly independent basis, but an **orthonormal** basis for the data. (An orthonormal basis is a basis <m>(\phi_1, \ldots \phi_n)</m> such that <m>\phi_i \cdot \phi_j = 0</m> if <m>i \ne j</m> and <m>1</m> if <m>i = j</m>).

Like sparse coding, independent component analysis has a simple mathematical formulation. Given some data <m>x</m>, we would like to learn a set of basis vectors which we represent in the columns of a matrix <m>W</m>, such that, firstly, as in sparse coding, our features are **sparse**; and secondly, our basis is an **orthonormal** basis. (Note that while in sparse coding, our matrix <m>A</m> was for mapping **features** <m>s</m> to **raw data**, in independent component analysis, our matrix <m>W</m> works in the opposite direction, mapping **raw data** <m>x</m> to **features** instead). This gives us the following objective function:

<m>
J(W) = \lVert Wx \rVert_1 
</m>

This objective function is equivalent to the sparsity penalty on the features <m>s</m> in sparse coding, since <m>Wx</m> is precisely the features that represent the data. Adding in the orthonormality constraint gives us the full optimization problem for independent component analysis:

<m>
\begin{array}{rcl}
     {\rm minimize} &amp; \lVert Wx \rVert_1  \\
     {\rm s.t.}     &amp; WW^T = I \\
\end{array} 
</m>

As is usually the case in deep learning, this problem has no simple analytic solution, and to make matters worse, the orthonormality constraint makes it slightly more difficult to optimize for the objective using gradient descent - every iteration of gradient descent must be followed by a step that maps the new basis back to the space of orthonormal bases (hence enforcing the constraint). 

In practice, optimizing for the objective function while enforcing the orthonormality constraint (as described in the section below) is feasible but slow. Hence, the use of orthonormal ICA is limited to situations where it is important to obtain an orthonormal basis.

### Orthonormal ICA ###

The orthonormal ICA objective is:

<m>
\begin{array}{rcl}
     {\rm minimize} &amp; \lVert Wx \rVert_1  \\
     {\rm s.t.}     &amp; WW^T = I
\end{array} 
</m>

Observe that the constraint <m>WW^T = I</m> implies two other constraints. 

Firstly, since we are learning an orthonormal basis, the number of basis vectors we learn must be less than the dimension of the input. In particular, this means that we cannot learn over-complete bases as we usually do in [[Sparse Coding: Autoencoder Interpretation | sparse coding]]. 

Secondly, the data must be [ZCA whitened]({{site.baseurl}}/unsupervised/PCAWhitening) with no regularization (that is, with <m>\epsilon</m> set to 0).
 
Hence, before we even begin to optimize for the orthonormal ICA objective, we must ensure that our data has been **whitened**, and that we are learning an **under-complete** basis. 

Following that, to optimize for the objective, we can use gradient descent, interspersing gradient descent steps with projection steps to enforce the orthonormality constraint. Hence, the procedure will be as follows:

Repeat until done:

<ol>
<li><m>W \leftarrow W - \alpha \nabla_W \lVert Wx \rVert_1</m></li>
<li><m>W \leftarrow \operatorname{proj}_U W</m> where <m>U</m> is the space of matrices satisfying <m>WW^T = I</m></li>
</ol>

In practice, the learning rate <m>\alpha</m> is varied using a line-search algorithm to speed up the descent, and the projection step is achieved by setting <m>W \leftarrow (WW^T)^{-\frac{1}{2}} W</m>, which can actually be seen as ZCA whitening (`TODO`: explain how it is like ZCA whitening).

