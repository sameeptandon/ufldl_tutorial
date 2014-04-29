---
layout: post
title:  "Sparse Coding"
date:   2013-08-28 13:32:12
categories: unsupervised
---
Sparse coding is a class of unsupervised methods for learning sets of over-complete bases to represent data efficiently. The aim of sparse coding is to find a set of basis vectors <m>\mathbf{\phi}_i</m> such that we can represent an input vector <m>\mathbf{x}</m> as a linear combination of these basis vectors:

<m>\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} 
\end{align}</m>

While techniques such as Principal Component Analysis (PCA) allow us to learn a complete set of basis vectors efficiently, we wish to learn an **over-complete** set of basis vectors to represent input vectors <m>\mathbf{x}\in\mathbb{R}^n</m> (i.e. such that <m>k > n</m>). The advantage of having an over-complete basis is that our basis vectors are better able to capture structures and patterns inherent in the input data. However, with an over-complete basis, the coefficients <m>a_i</m> are no longer uniquely determined by the input vector <m>\mathbf{x}</m>. Therefore, in sparse coding, we introduce the additional criterion of **sparsity** to resolve the degeneracy introduced by over-completeness. 

Here, we define sparsity as having few non-zero components or having few components not close to zero. The requirement that our coefficients <m>a_i</m> be sparse means that given a input vector, we would like as few of our coefficients to be far from zero as possible. The choice of sparsity as a desired characteristic of our representation of the input data can be motivated by the observation that most sensory data such as natural images may be described as the superposition of a small number of atomic elements such as surfaces or edges. Other justifications such as comparisons to the properties of the primary visual cortex have also been advanced. 

We define the sparse coding cost function on a set of <m>m</m> input vectors as

<m>\begin{align}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i)
\end{align}</m>

where <m>S(.)</m> is a sparsity cost function which penalizes <m>a_i</m> for being far from zero. We can interpret the first term of the sparse coding objective as a reconstruction term which tries to force the algorithm to provide a good representation of <m>\mathbf{x}</m> and the second term as a sparsity penalty which forces our representation of <m>\mathbf{x}</m> to be sparse. The constant <m>\lambda</m> is a scaling constant to determine the relative importance of these two contributions. 

Although the most direct measure of sparsity is the "<m>L_0</m>" norm (<m>S(a_i) = \mathbf{1}(|a_i|>0)</m>), it is non-differentiable and difficult to optimize in general. In practice, common choices for the sparsity cost <m>S(.)</m> are the <m>L_1</m> penalty <m>S(a_i)=\left|a_i\right|_1 </m> and the log penalty <m>S(a_i)=\log(1+a_i^2)</m>.

In addition, it is also possible to make the sparsity penalty arbitrarily small by scaling down <m>a_i</m> and scaling <m>\mathbf{\phi}_i</m> up by some large constant. To prevent this from happening, we will constrain <m>\left|\left|\mathbf{\phi}\right|\right|^2</m> to be less than some constant <m>C</m>. The full sparse coding cost function including our constraint on <m>\mathbf{\phi}</m> is

<m>\begin{array}{rc}
\text{minimize}_{a^{(j)}_i,\mathbf{\phi}_{i}} &amp; \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
\\
\text{subject to}  &amp;  \left|\left|\mathbf{\phi}_i\right|\right|^2 \leq C, \forall i = 1,...,k 
\\
\end{array}</m>

### Probabilistic Interpretation ###
So far, we have considered sparse coding in the context of finding a sparse, over-complete set of basis vectors to span our input space. Alternatively, we may also approach sparse coding from a probabilistic perspective as a generative model. 

Consider the problem of modelling natural images as the linear superposition of <m>k</m> independent source features <m>\mathbf{\phi}_i</m> with some additive noise <m>\nu</m>:

<m>\begin{align}
\mathbf{x} = \sum_{i=1}^k a_i \mathbf{\phi}_{i} + \nu(\mathbf{x})
\end{align}</m>

Our goal is to find a set of basis feature vectors <m>\mathbf{\phi}</m> such that the distribution of images <m>P(\mathbf{x}\mid\mathbf{\phi})</m> is as close as possible to the empirical distribution of our input data <m>P^*(\mathbf{x})</m>. One method of doing so is to minimize the KL divergence between <m>P^*(\mathbf{x})</m> and <m>P(\mathbf{x}\mid\mathbf{\phi})</m> where the KL divergence is defined as:

<m>\begin{align}
D(P^*(\mathbf{x})||P(\mathbf{x}\mid\mathbf{\phi})) = \int P^*(\mathbf{x}) \log \left(\frac{P^*(\mathbf{x})}{P(\mathbf{x}\mid\mathbf{\phi})}\right)d\mathbf{x}
\end{align}</m> 

Since the empirical distribution <m>P^*(\mathbf{x})</m> is constant across our choice of <m>\mathbf{\phi}</m>, this is equivalent to maximizing the log-likelihood of <m>P(\mathbf{x}\mid\mathbf{\phi})</m>.

Assuming <m>\nu</m> is Gaussian white noise with variance <m>\sigma^2</m>, we have that

<m>\begin{align}
P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) = \frac{1}{Z} \exp\left(- \frac{(\mathbf{x}-\sum^{k}_{i=1} a_i \mathbf{\phi}_{i})^2}{2\sigma^2}\right)
\end{align}</m>

In order to determine the distribution <m>P(\mathbf{x}\mid\mathbf{\phi})</m>, we also need to specify the prior distribution <m>P(\mathbf{a})</m>. Assuming the independence of our source features, we can factorize our prior probability as 

<m>\begin{align}
P(\mathbf{a}) = \prod_{i=1}^{k} P(a_i)
\end{align}</m>

At this point, we would like to incorporate our sparsity assumption -- the assumption that any single image is likely to be the product of relatively few source features. Therefore, we would like the probability distribution of <m>a_i</m> to be peaked at zero and have high kurtosis. A convenient parameterization of the prior distribution is 

<m>\begin{align}
P(a_i) = \frac{1}{Z}\exp(-\beta S(a_i))
\end{align}</m>

Where <m>S(a_i)</m> is a function determining the shape of the prior distribution.

Having defined <m>P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi})</m> and <m> P(\mathbf{a})</m>, we can write the probability of the data <m>\mathbf{x}</m> under the model defined by <m>\mathbf{\phi}</m> as 

<m>\begin{align}
P(\mathbf{x} \mid \mathbf{\phi}) = \int P(\mathbf{x} \mid \mathbf{a}, \mathbf{\phi}) P(\mathbf{a}) d\mathbf{a}
\end{align}</m>

and our problem reduces to finding

<m>
\begin{align}
\mathbf{\phi}^*=\text{argmax}_{\mathbf{\phi}} E\left[ \log(P(\mathbf{x} \mid \mathbf{\phi})) \right]
\end{align}
</m>

Where <m>E\left[ \cdot \right]</m> denotes expectation over our input data. 

Unfortunately, the integral over <m>\mathbf{a}</m> to obtain <m>P(\mathbf{x} \mid \mathbf{\phi})</m> is generally intractable. We note though that if the distribution of <m>P(\mathbf{x} \mid \mathbf{\phi})</m> is sufficiently peaked (w.r.t. <m>\mathbf{a}</m>), we can approximate its integral with the maximum value of  <m>P(\mathbf{x} \mid \mathbf{\phi})</m> and obtain a approximate solution 

<m>
\begin{align}
\mathbf{\phi}^{*'}=\text{argmax}_{\mathbf{\phi}} E\left[ \max_{\mathbf{a}} \log(P(\mathbf{x} \mid \mathbf{\phi})) \right]
\end{align}
</m>

As before, we may increase the estimated probability by scaling down <m>a_i</m> and scaling up <m>\mathbf{\phi}</m> (since <m>P(a_i)</m> peaks about zero) , we therefore impose a norm constraint on our features <m>\mathbf{\phi}</m> to prevent this.

Finally, we can recover our original cost function by defining the energy function of this linear generative model

<m>\begin{array}{rl}
E\left( \mathbf{x} , \mathbf{a} \mid \mathbf{\phi} \right) &amp; := -\log \left( P(\mathbf{x}\mid \mathbf{\phi},\mathbf{a}\right)P(\mathbf{a})) \\
 &amp;= \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
 \end{array}</m>

 where <m>\lambda = 2\sigma^2\beta</m> and irrelevant constants have been hidden. Since maximizing the log-likelihood is equivalent to minimizing the energy function, we recover the original optimization problem:
 
 <m>\begin{align}
 \mathbf{\phi}^{*},\mathbf{a}^{*}=\text{argmin}_{\mathbf{\phi},\mathbf{a}} \sum_{j=1}^{m} \left|\left| \mathbf{x}^{(j)} - \sum_{i=1}^k a^{(j)}_i \mathbf{\phi}_{i}\right|\right|^{2} + \lambda \sum_{i=1}^{k}S(a^{(j)}_i) 
 \end{align}</m>

 Using a probabilistic approach, it can also be seen that the choices of the <m>L_1</m> penalty <m>\left|a_i\right|_1 </m> and the log penalty <m>\log(1+a_i^2)</m> for <m>S(.)</m> correspond to the use of the Laplacian <m>P(a_i) \propto \exp\left(-\beta|a_i|\right)</m> and the Cauchy prior <m>P(a_i) \propto \frac{\beta}{1+a_i^2}</m> respectively.

### Learning ###
 Learning a set of basis vectors <m>\mathbf{\phi}</m> using sparse coding consists of performing two separate optimizations, the first being an optimization over coefficients <m>a_i</m> for each training example <m>\mathbf{x}</m> and the second an optimization over basis vectors <m>\mathbf{\phi}</m> across many training examples at once.

 Assuming an <m>L_1</m> sparsity penalty, learning <m>a^{(j)}_i</m> reduces to solving a <m>L_1</m> regularized least squares problem which is convex in <m>a^{(j)}_i</m> for which several techniques have been developed (convex optimization software such as CVX can also be used to perform L1 regularized least squares). Assuming a differentiable <m>S(.)</m> such as the log penalty, gradient-based methods such as conjugate gradient methods can also be used.

 Learning a set of basis vectors with a <m>L_2</m> norm constraint also reduces to a least squares problem with quadratic constraints which is convex in <m>\mathbf{\phi}</m>. Standard convex optimization software (e.g. CVX) or other iterative methods can be used to solve for <m>\mathbf{\phi}</m> although significantly more efficient methods such as solving the Lagrange dual have also been developed.

 As described above, a significant limitation of sparse coding is that even after a set of basis vectors have been learnt, in order to "encode" a new data example, optimization must be performed to obtain the required coefficients. This significant "runtime" cost means that sparse coding is computationally expensive to implement even at test time especially compared to typical feedforward architectures.
