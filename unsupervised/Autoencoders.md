---
layout: post
title:  "Autoencoders"
date:   2013-08-28 13:32:12
categories: unsupervised
---

So far, we have described the application of neural networks to supervised learning, in which we have labeled
training examples.  Now suppose we have only a set of unlabeled training examples <m>\textstyle \{x^{(1)}, x^{(2)}, x^{(3)}, \ldots\}</m>,
where <m>\textstyle x^{(i)} \in \Re^{n}</m>.  An
**autoencoder** neural network is an unsupervised learning algorithm that applies backpropagation,
setting the target values to be equal to the inputs.  I.e., it uses <m>\textstyle y^{(i)} = x^{(i)}</m>.

Here is an autoencoder:

<img src="{{site.baseurl}}/images/Autoencoder636.png" width="100%">

The autoencoder tries to learn a function <m>\textstyle h_{W,b}(x) \approx x</m>.  In other
words, it is trying to learn an approximation to the identity function, so as
to output <m>\textstyle \hat{x}</m> that is similar to <m>\textstyle x</m>.  The identity function seems a
particularly trivial function to be trying to learn; but by placing constraints
on the network, such as by limiting the number of hidden units, we can discover
interesting structure about the data.  As a concrete example, suppose the
inputs <m>\textstyle x</m> are the pixel intensity values from a <m>\textstyle 10 \times 10</m> image (100
pixels) so <m>\textstyle n=100</m>, and there are <m>\textstyle s_2=50</m> hidden units in layer <m>\textstyle L_2</m>.  Note that
we also have <m>\textstyle y \in \Re^{100}</m>.  Since there are only 50 hidden units, the
network is forced to learn a ''compressed'' representation of the input.
I.e., given only the vector of hidden unit activations <m>\textstyle a^{(2)} \in \Re^{50}</m>,
it must try to '''reconstruct''' the 100-pixel input <m>\textstyle x</m>.  If the input were completely
random---say, each <m>\textstyle x_i</m> comes from an IID Gaussian independent of the other
features---then this compression task would be very difficult.  But if there is
structure in the data, for example, if some of the input features are correlated,
then this algorithm will be able to discover some of those correlations. In fact,
this simple autoencoder often ends up learning a low-dimensional representation very similar
to PCAs.

Our argument above relied on the number of hidden units <m>\textstyle s_2</m> being small.  But
even when the number of hidden units is large (perhaps even greater than the
number of input pixels), we can still discover interesting structure, by
imposing other constraints on the network.  In particular, if we impose a
'''sparsity''' constraint on the hidden units, then the autoencoder will still
discover interesting structure in the data, even if the number of hidden units
is large.

Informally, we will think of a neuron as being "active" (or as "firing") if
its output value is close to 1, or as being "inactive" if its output value is
close to 0.  We would like to constrain the neurons to be inactive most of the
time. This discussion assumes a sigmoid activation function.  If you are
using a tanh activation function, then we think of a neuron as being inactive
when it outputs values close to -1.

Recall that <m>\textstyle a^{(2)}_j</m> denotes the activation of hidden unit <m>\textstyle j</m> in the
autoencoder.  However, this notation doesn't make explicit what was the input <m>\textstyle x</m>
that led to that activation.   Thus, we will write <m>\textstyle a^{(2)}_j(x)</m> to denote the activation
of this hidden unit when the network is given a specific input <m>\textstyle x</m>.  

Further, let

<m>\begin{align}
\hat\rho_j = \frac{1}{m} \sum_{i=1}^m \left[ a^{(2)}_j(x^{(i)}) \right]
\end{align}</m>

be the average activation of hidden unit <m>\textstyle j</m> (averaged over the training set).
We would like to (approximately) enforce the constraint

<m>\begin{align}
\hat\rho_j = \rho,
\end{align}</m>

where <m>\textstyle \rho</m> is a '''sparsity parameter''', typically a small value close to zero
(say <m>\textstyle \rho = 0.05</m>).  In other words, we would like the average activation
of each hidden neuron <m>\textstyle j</m> to be close to 0.05 (say).  To satisfy this
constraint, the hidden unit's activations must mostly be near 0.

To achieve this, we will add an extra penalty term to our optimization objective that
penalizes <m>\textstyle \hat\rho_j</m> deviating significantly from <m>\textstyle \rho</m>.  Many choices of the penalty
term will give reasonable results.  We will choose the following:

<m>\begin{align}
\sum_{j=1}^{s_2} \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}.
\end{align}</m>

Here, <m>\textstyle s_2</m> is the number of neurons in the hidden layer, and the index <m>\textstyle j</m> is summing
over the hidden units in our network.  If you are
familiar with the concept of KL divergence, this penalty term is based on
it, and can also be written

<m>\begin{align}
\sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}</m>

where <m>\textstyle {\rm KL}(\rho || \hat\rho_j)
 = \rho \log \frac{\rho}{\hat\rho_j} + (1-\rho) \log \frac{1-\rho}{1-\hat\rho_j}</m>
 is the Kullback-Leibler (KL) divergence between
 a Bernoulli random variable with mean <m>\textstyle \rho</m> and a Bernoulli random variable with mean <m>\textstyle \hat\rho_j</m>.
 KL-divergence is a standard function for measuring how different two different
 distributions are.  (If you've not seen KL-divergence before, don't worry about
 it; everything you need to know about it is contained in these notes.)

 This penalty function has the property that <m>\textstyle {\rm KL}(\rho || \hat\rho_j) = 0</m> if <m>\textstyle \hat\rho_j = \rho</m>,
 and otherwise it increases monotonically as <m>\textstyle \hat\rho_j</m> diverges from <m>\textstyle \rho</m>.  For example, in the
 figure below, we have set <m>\textstyle \rho = 0.2</m>, and plotted <m>\textstyle {\rm KL}(\rho || \hat\rho_j)</m> for a range of values of <m>\textstyle \hat\rho_j</m>:

<img src="{{site.baseurl}}/images/KLPenaltyExample.png" width="100%">

We see that the KL-divergence reaches its minimum of 0 at
<m>\textstyle \hat\rho_j = \rho</m>, and blows up (it actually
approaches <m>\textstyle \infty</m>) as <m>\textstyle
\hat\rho_j</m> approaches 0 or 1.  Thus, minimizing this penalty
term has the effect of causing <m>\textstyle \hat\rho_j</m> to be
close to <m>\textstyle \rho</m>.

Our overall cost function is now

<m>\begin{align}
J_{\rm sparse}(W,b) = J(W,b) + \beta \sum_{j=1}^{s_2} {\rm KL}(\rho || \hat\rho_j),
\end{align}</m>

where <m>\textstyle J(W,b)</m> is as defined previously, and <m>\textstyle \beta</m> controls the weight of
the sparsity penalty term.  The term <m>\textstyle \hat\rho_j</m> (implicitly) depends on <m>\textstyle W,b</m> also,
because it is the average activation of hidden unit <m>\textstyle j</m>, and the activation of a hidden
unit depends on the parameters <m>\textstyle W,b</m>.

To incorporate the KL-divergence term into your derivative calculation, there is a simple-to-implement
trick involving only a small change to your code.  Specifically, where previously for
the second layer (<m>\textstyle l=2</m>), during backpropagation you would have computed

<m>\begin{align}
\delta^{(2)}_i = \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right) f'(z^{(2)}_i),
\end{align}</m>

now instead compute

<m>\begin{align}
\delta^{(2)}_i =
  \left( \left( \sum_{j=1}^{s_{2}} W^{(2)}_{ji} \delta^{(3)}_j \right)
+ \beta \left( - \frac{\rho}{\hat\rho_i} + \frac{1-\rho}{1-\hat\rho_i} \right) \right) f'(z^{(2)}_i) .
\end{align}</m>

One subtlety is that you'll need to know <m>\textstyle \hat\rho_i</m> to compute this term.  Thus, you'll need
to compute a forward pass on all the training examples first to compute the average
activations on the training set, before computing backpropagation on any example.  If your
training set is small enough to fit comfortably in computer memory (this will be the case for the programming
assignment), you can compute forward passes on all your examples and keep the resulting activations
in memory and compute the <m>\textstyle \hat\rho_i</m>s.  Then you can use your precomputed activations to
perform backpropagation on all your examples.  If your data is too large to fit in memory, you
may have to scan through your examples computing a forward pass on each to accumulate (sum up) the
activations and compute <m>\textstyle \hat\rho_i</m> (discarding the result of each forward pass after you
have taken its activations <m>\textstyle a^{(2)}_i</m> into account for computing <m>\textstyle \hat\rho_i</m>).  Then after
having computed <m>\textstyle \hat\rho_i</m>, you'd have to redo the forward pass for each example so that you
can do backpropagation on that example.  In this latter case, you would end up computing a forward
pass twice on each example in your training set, making it computationally less efficient.

The full derivation showing that the algorithm above results in
gradient descent is beyond the scope of these notes.  But if you
implement the autoencoder using backpropagation modified this way,
you will be performing gradient descent exactly on the objective <m>\textstyle J_{\rm sparse}(W,b)</m>. Using the derivative
checking method, you will be able to verify this for yourself as
well.

### Visualizing a Trained Autoencoder ###

Having trained a (sparse) autoencoder, we would now like to visualize the function
learned by the algorithm, to try to understand what it has learned.
Consider the case of training an autoencoder on <m>\textstyle 10 \times 10</m> images, so that <m>\textstyle n = 100</m>.
Each hidden unit <m>\textstyle i</m> computes a function of the input:

<m>\begin{align}
a^{(2)}_i = f\left(\sum_{j=1}^{100} W^{(1)}_{ij} x_j  + b^{(1)}_i \right).
\end{align}</m>

We will visualize the function computed by hidden unit <m>\textstyle i</m>---which depends on the
parameters <m>\textstyle W^{(1)}_{ij}</m> (ignoring
the bias term for now)---using a 2D image.  In particular, we
think of <m>\textstyle a^{(2)}_i</m> as some non-linear feature of
the input <m>\textstyle x</m>.
We ask: What input image <m>\textstyle x</m> would cause <m>\textstyle a^{(2)}_i</m> to be maximally activated?
(Less formally, what is the feature that hidden unit <m>\textstyle i</m> is looking for?)
For this question to have a non-trivial answer,
we must impose some constraints on <m>\textstyle x</m>.  If we suppose that
the input is
norm constrained by <m>\textstyle ||x||^2 = \sum_{i=1}^{100} x_i^2 \leq 1</m>, then one can
show (try doing this yourself)
that the input which maximally activates hidden unit <m>\textstyle i</m> is given
by setting pixel <m>\textstyle x_j</m> (for all 100 pixels, <m>\textstyle j=1,\ldots, 100</m>) to

<m>\begin{align}
x_j = \frac{W^{(1)}_{ij}}{\sqrt{\sum_{j=1}^{100} (W^{(1)}_{ij})^2}}.
\end{align}</m>

By displaying the image formed by these pixel intensity values, we can begin
to understand what feature hidden unit <m>\textstyle i</m> is looking for.

If we have an autoencoder with 100 hidden units (say), then we our
visualization will have 100 such images---one per hidden unit.  By examining
these 100 images, we can try to understand what the ensemble of hidden units is
learning.

When we do this for a sparse autoencoder (trained with 100 hidden units on
10x10 pixel inputs<sup>1</sup> we get the following result:

<img src="{{site.baseurl}}/images/ExampleSparseAutoencoderWeights.png" width="400px">

Each square in the figure above shows the (norm bounded) input image <m>\textstyle x</m> that
maximally actives one of 100 hidden units.  We see that the different hidden
units have learned to detect edges at different positions and orientations in
the image.

These features are, not surprisingly, useful for such tasks as object
recognition and other vision tasks.  When applied to other input domains (such
as audio), this algorithm also learns useful representations/features for those
domains too.

----

<sup>1</sup> ''The learned features were obtained by training on '''whitened''' natural images.  Whitening is a preprocessing step which removes redundancy in the input, by causing adjacent pixels to become less correlated.''
