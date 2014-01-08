---
layout: post
title:  "Multi-Layer Neural Network"
date:   2013-08-28 13:32:12
categories: supervised
---
Consider a supervised learning problem where we have access to labeled training
examples <m>(x^{(i)}, y^{(i)})</m>.  Neural networks give a way of defining a complex,
non-linear form of hypotheses <m>h_{W,b}(x)</m>, with parameters <m>W,b</m> that we can
fit to our data.

To describe neural networks, we will begin by describing the simplest possible
neural network, one which comprises a single "neuron."  We will use the following
diagram to denote a single neuron:

<center>
<img src="{{site.baseurl}}/images/SingleNeuron.png" width="300">
</center>

This "neuron" is a computational unit that takes as input <m>x_1, x_2, x_3</m> (and a +1 intercept term), and outputs <m>\textstyle h_{W,b}(x) = f(W^Tx) = f(\sum_{i=1}^3 W_{i}x_i +b)</m>, where <m>f : \Re \mapsto \Re</m> is called the __activation function__.  In these notes, we will choose <m>f(\cdot)</m> to be the sigmoid function:

<m>
f(z) = \frac{1}{1+\exp(-z)}.
</m>

Thus, our single neuron corresponds exactly to the input-output mapping defined by logistic regression.

Although these notes will use the sigmoid function, it is worth noting that
another common choice for <m>f</m> is the hyperbolic tangent, or tanh, function:

<m>
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}.
</m>

Recent research has found a different activation function, the rectified linear function, often works better in practice for deep neural networks. This activation function is different from sigmoid and <m>\tanh</m> because it is not bounded or continuously differentiable. The rectified linear activation function is given by,

<m>
f(z) = \max(0,x). 
</m>

Here are plots of the sigmoid, <m>\tanh</m> and rectified linear functions:

<center>
<img src="{{site.baseurl}}/images/Activation_functions.png" width="450">
</center>

The <m>\tanh(z)</m> function is a rescaled version of the sigmoid, and its output range is <m>[-1,1]</m> instead of <m>[0,1]</m>. The rectified linear function is piece-wise linear and saturates at exactly 0 whenever the input <m>z</m> is less than 0. 

Note that unlike some other venues (including the OpenClassroom videos, and parts of CS229),  we are not using the convention here of <m>x_0=1</m>.  Instead, the intercept term is handled separately by the parameter <m>b</m>.

Finally, one identity that'll be useful later: If <m>f(z) = 1/(1+\exp(-z))</m> is the sigmoid function, then its derivative is given by <m>f'(z) = f(z) (1-f(z))</m>. (If <m>f</m> is the tanh function, then its derivative is given by <m>f'(z) = 1- (f(z))^2</m>.)  You can derive this yourself using the definition of the sigmoid (or tanh) function. The rectified linear function has gradient 0 when <m>z \leq 0</m> and 1 otherwise. The gradient is undefined at <m>z=0</m>, though this doesn't cause problems in practice because we average the gradient over many training examples during optimization. 


### Neural Network model ###

A neural network is put together by hooking together many of our simple
"neurons," so that the output of a neuron can be the input of another.  For
example, here is a small neural network:

<center>
<img src="{{site.baseurl}}/images/Network331.png" width="400">
</center>

In this figure, we have used circles to also denote the inputs to the network.  The circles
labeled "+1" are called __bias units__, and correspond to the intercept term.
The leftmost layer of the network is called the __input layer__, and the
rightmost layer the __output layer__ (which, in this example, has only one
node).  The middle layer of nodes is called the __hidden layer__, because its
values are not observed in the training set.  We also say that our example
neural network has 3 __input units__ (not counting the bias unit), 3 
__hidden units__, and 1 __output unit__.

We will let <m>n_l</m> denote the number of layers in our network; thus <m>n_l=3</m> in our example.  We label layer <m>l</m> as <m>L_l</m>, so layer <m>L_1</m> is the input layer, and layer <m>L_{n_l}</m> the output layer.  Our neural network has parameters <m>(W,b) = (W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)})</m>, where we write <m>W^{(l)}_{ij}</m> to denote the parameter (or weight) associated with the connection between unit <m>j</m> in layer <m>l</m>, and unit <m>i</m> in layer <m>l+1</m>.  (Note the order of the indices.) Also, <m>b^{(l)}_i</m> is the bias associated with unit <m>i</m> in layer <m>l+1</m>. Thus, in our example, we have <m>W^{(1)} \in \Re^{3\times 3}</m>, and <m>W^{(2)} \in \Re^{1\times 3}</m>. Note that bias units don't have inputs or connections going into them, since they always output the value +1.  We also let <m>s_l</m> denote the number of nodes in layer <m>l</m> (not counting the bias unit).

We will write <m>a^{(l)}_i</m> to denote the __activation__ (meaning output value) of
unit <m>i</m> in layer <m>l</m>.  For <m>l=1</m>, we also use <m>a^{(1)}_i = x_i</m> to denote the <m>i</m>-th input. Given a fixed setting of the parameters <m>W,b</m>, our neural network defines a hypothesis <m>h_{W,b}(x)</m> that outputs a real number.  Specifically, the computation that this neural network represents is given by:

<m>
\begin{align}
a_1^{(2)} &amp;= f(W_{11}^{(1)}x_1 + W_{12}^{(1)} x_2 + W_{13}^{(1)} x_3 + b_1^{(1)})  \\
a_2^{(2)} &amp;= f(W_{21}^{(1)}x_1 + W_{22}^{(1)} x_2 + W_{23}^{(1)} x_3 + b_2^{(1)})  \\
a_3^{(2)} &amp;= f(W_{31}^{(1)}x_1 + W_{32}^{(1)} x_2 + W_{33}^{(1)} x_3 + b_3^{(1)})  \\
h_{W,b}(x) &amp;= a_1^{(3)} =  f(W_{11}^{(2)}a_1^{(2)} + W_{12}^{(2)} a_2^{(2)} + W_{13}^{(2)} a_3^{(2)} + b_1^{(2)}) 
\end{align}
</m>

In the sequel, we also let <m>z^{(l)}_i</m> denote the total weighted sum of inputs to unit <m>i</m> in layer <m>l</m>, including the bias term (e.g., <m>\textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i</m>), so that <m>a^{(l)}_i = f(z^{(l)}_i)</m>.

Note that this easily lends itself to a more compact notation.  Specifically, if we extend the
activation function <m>f(\cdot)</m>
to apply to vectors in an element-wise fashion (i.e., <m>f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)]</m>), then we can write the equations above more compactly as:

<m>\begin{align}
z^{(2)} &amp;= W^{(1)} x + b^{(1)} \\
a^{(2)} &amp;= f(z^{(2)}) \\
z^{(3)} &amp;= W^{(2)} a^{(2)} + b^{(2)} \\
h_{W,b}(x) &amp;= a^{(3)} = f(z^{(3)})
\end{align}
</m>

We call this step __forward propagation.__  More generally, recalling that we also use <m>a^{(1)} = x</m> to also denote the values from the input layer, then given layer <m>l</m>'s activations <m>a^{(l)}</m>, we can compute layer <m>l+1</m>'s activations <m>a^{(l+1)}</m> as:

<m>
\begin{align}
z^{(l+1)} &amp;= W^{(l)} a^{(l)} + b^{(l)}   \\
a^{(l+1)} &amp;= f(z^{(l+1)})
\end{align}
</m>

By organizing our parameters in matrices and using matrix-vector operations, we can take
advantage of fast linear algebra routines to quickly perform calculations in our network.


We have so far focused on one example neural network, but one can also build neural
networks with other __architectures__ (meaning patterns of connectivity between neurons), including ones with multiple hidden layers.
The most common choice is a <m>\textstyle n_l</m>-layered network
where layer <m>\textstyle 1</m> is the input layer, layer <m>\textstyle n_l</m> is the output layer, and each
layer <m>\textstyle l</m> is densely connected to layer <m>\textstyle l+1</m>.  In this setting, to compute the
output of the network, we can successively compute all the activations in layer <m>\textstyle L_2</m>, then layer <m>\textstyle L_3</m>, and so on, up to layer <m>\textstyle L_{n_l}</m>, using the equations above that describe the forward propagation step.  This is one example of a __feedforward__ neural network, since the connectivity graph does not have any directed loops or cycles.


Neural networks can also have multiple output units.  For example, here is a network
with two hidden layers layers <m>L_2</m> and <m>L_3</m> and two output units in layer <m>L_4</m>:

<center>
<img src="{{site.baseurl}}/images/Network3322.png" width="500">
</center>

To train this network, we would need training examples <m>(x^{(i)}, y^{(i)})</m>
where <m>y^{(i)} \in \Re^2</m>.  This sort of network is useful if there're multiple
outputs that you're interested in predicting.  (For example, in a medical
diagnosis application, the vector <m>x</m> might give the input features of a
patient, and the different outputs <m>y_i</m>'s might indicate presence or absence
of different diseases.)



### Backpropagation Algorithm ###


Suppose we have a fixed training set <m>\{ (x^{(1)}, y^{(1)}), \ldots, (x^{(m)}, y^{(m)}) \}</m> of <m>m</m> training examples. We can train our neural network using batch gradient descent.  In detail, for a single training example <m>(x,y)</m>, we define the cost function with respect to that single example to be:

<m>
\begin{align}
J(W,b; x,y) = \frac{1}{2} \left\| h_{W,b}(x) - y \right\|^2.
\end{align}
</m>

This is a (one-half) squared-error cost function. Given a training set of <m>m</m> examples, we then define the overall cost function to be: 

<m>
\begin{align}
J(W,b)
&amp;= \left[ \frac{1}{m} \sum_{i=1}^m J(W,b;x^{(i)},y^{(i)}) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
 \\
&amp;= \left[ \frac{1}{m} \sum_{i=1}^m \left( \frac{1}{2} \left\| h_{W,b}(x^{(i)}) - y^{(i)} \right\|^2 \right) \right]
                       + \frac{\lambda}{2} \sum_{l=1}^{n_l-1} \; \sum_{i=1}^{s_l} \; \sum_{j=1}^{s_{l+1}} \left( W^{(l)}_{ji} \right)^2
\end{align}
</m>

The first term in the definition of <m>J(W,b)</m> is an average sum-of-squares error term. The second term is a regularization term (also called a __weight decay__ term) that tends to decrease the magnitude of the weights, and helps prevent overfitting.

(Note: Usually weight decay is not applied to the bias terms <m>b^{(l)}_i</m>, as reflected in our definition for <m>J(W, b)</m>.  Applying weight decay to the bias units usually makes only a small difference to the final network, however.  If you've taken CS229 (Machine Learning) at Stanford or watched the course's videos on YouTube, you may also recognize this weight decay as essentially a variant of the Bayesian regularization method you saw there, where we placed a Gaussian prior on the parameters and did MAP (instead of maximum likelihood) estimation.)

The __weight decay parameter__ <m>\lambda</m> controls the relative importance of the two terms. Note also the slightly overloaded notation: <m>J(W,b;x,y)</m> is the squared error cost with respect to a single example; <m>J(W,b)</m> is the overall cost function, which includes the weight decay term.

This cost function above is often used both for classification and for regression problems. For classification, we let <m>y = 0</m> or <m>1</m> represent the two class labels (recall that the sigmoid activation function outputs values in <m>[0,1]</m>; if we were using a tanh activation function, we would instead use -1 and +1 to denote the labels).  For regression problems, we first scale our outputs to ensure that they lie in the <m>[0,1]</m> range (or if we were using a tanh activation function, then the <m>[-1,1]</m> range).

Our goal is to minimize <m>J(W,b)</m> as a function of <m>W</m> and <m>b</m>. To train our neural network, we will initialize each parameter <m>W^{(l)}_{ij}</m> and each <m>b^{(l)}_i</m> to a small random value near zero (say according to a <m>{Normal}(0,\epsilon^2)</m> distribution for some small <m>\epsilon</m>, say <m>0.01</m>), and then apply an optimization algorithm such as batch gradient descent. Since <m>J(W, b)</m> is a non-convex function,
gradient descent is susceptible to local optima; however, in practice gradient descent
usually works fairly well. Finally, note that it is important to initialize
the parameters randomly, rather than to all 0's.  If all the parameters start off
at identical values, then all the hidden layer units will end up learning the same
function of the input (more formally, <m>W^{(1)}_{ij}</m> will be the same for all values of <m>i</m>, so that <m>a^{(2)}_1 = a^{(2)}_2 = a^{(2)}_3 = \ldots</m> for any input <m>x</m>). The random initialization serves the purpose of __symmetry breaking__.

One iteration of gradient descent updates the parameters <m>W,b</m> as follows:

<m>
\begin{align}
W_{ij}^{(l)} &amp;= W_{ij}^{(l)} - \alpha \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) \\
b_{i}^{(l)} &amp;= b_{i}^{(l)} - \alpha \frac{\partial}{\partial b_{i}^{(l)}} J(W,b)
\end{align}
</m>

where <m>\alpha</m> is the learning rate.  The key step is computing the partial derivatives above. We will now describe the __backpropagation__ algorithm, which gives an
efficient way to compute these partial derivatives.

We will first describe how backpropagation can be used to compute <m>\textstyle \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y)</m> and <m>\textstyle \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y)</m>, the partial derivatives of the cost function <m>J(W,b;x,y)</m> defined with respect to a single example <m>(x,y)</m>. Once we can compute these, we see that the derivative of the overall cost function <m>J(W,b)</m> can be computed as:

<m>
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b) &amp;=
\left[ \frac{1}{m} \sum_{i=1}^m \frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x^{(i)}, y^{(i)}) \right] + \lambda W_{ij}^{(l)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b) &amp;=
\frac{1}{m}\sum_{i=1}^m \frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x^{(i)}, y^{(i)})
\end{align}
</m>

The two lines above differ slightly because weight decay is applied to <m>W</m> but not <m>b</m>.

The intuition behind the backpropagation algorithm is as follows. Given a training example <m>(x,y)</m>, we will first run a "forward pass" to compute all the activations throughout the network, including the output value of the hypothesis <m>h_{W,b}(x)</m>.  Then, for each node <m>i</m> in layer <m>l</m>, we would like to compute an "error term" <m>\delta^{(l)}_i</m> that measures how much that node was "responsible" for any errors in our output. For an output node, we can directly measure the difference between the network's activation and the true target value, and use that to define <m>\delta^{(n_l)}_i</m> (where layer <m>n_l</m> is the output layer).  How about hidden units?  For those, we will compute <m>\delta^{(l)}_i</m> based on a weighted average of the error terms of the nodes that uses <m>a^{(l)}_i</m> as an input.  In detail, here is the backpropagation algorithm:


1. Perform a feedforward pass, computing the activations for layers <m>L_2</m>, <m>L_3</m>, and so on up to the output layer <m>L_{n_l}</m>.
2. For each output unit <m>i</m> in layer <m>n_l</m> (the output layer), set
	
	<m>
	\begin{align}
	\delta^{(n_l)}_i
	= \frac{\partial}{\partial z^{(n_l)}_i} \;\;
	\frac{1}{2} \left\|y - h_{W,b}(x)\right\|^2 = - (y_i - a^{(n_l)}_i) \cdot f'(z^{(n_l)}_i)
	\end{align}
	</m>
3. For <m>l = n_l-1, n_l-2, n_l-3, \ldots, 2</m> 

	&nbsp;&nbsp;&nbsp;&nbsp;For each node <m>i</m> in layer <m>l</m>, set
	
	&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<m>
	\delta^{(l)}_i = \left( \sum_{j=1}^{s_{l+1}} W^{(l)}_{ji} \delta^{(l+1)}_j \right) f'(z^{(l)}_i)
	</m>
4. Compute the desired partial derivatives, which are given as: 

<m>
\begin{align}
\frac{\partial}{\partial W_{ij}^{(l)}} J(W,b; x, y) &amp;= a^{(l)}_j \delta_i^{(l+1)} \\
\frac{\partial}{\partial b_{i}^{(l)}} J(W,b; x, y) &amp;= \delta_i^{(l+1)}.
\end{align}
</m>


Finally, we can also re-write the algorithm using matrix-vectorial notation. We will use "<m>\textstyle \bullet</m>" to denote the element-wise product operator (denoted `.*` in Matlab or Octave, and also called the Hadamard product), so that if <m>\textstyle a = b \bullet c</m>, then <m>\textstyle a_i = b_ic_i</m>. Similar to how we extended the definition of <m>\textstyle f(\cdot)</m> to apply element-wise to vectors, we also do the same for <m>\textstyle f'(\cdot)</m> (so that <m> \textstyle f'([z_1, z_2, z_3]) = [f'(z_1), f'(z_2), f'(z_3)]</m>).

The algorithm can then be written:

1. Perform a feedforward pass, computing the activations for layers <m>\textstyle L_2</m>, <m>\textstyle L_3</m>, up to the output layer <m>\textstyle L_{n_l}</m>, using the equations defining the forward propagation steps
2. For the output layer (layer <m>\textstyle n_l</m>), set

	<m>\begin{align} \delta^{(n_l)} = - (y - a^{(n_l)}) \bullet f'(z^{(n_l)}) \end{align}</m>
3. For <m>\textstyle l = n_l-1, n_l-2, n_l-3, \ldots, 2</m>, set

  	<m>\begin{align} \delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)}) \end{align}</m>
4. Compute the desired partial derivatives: 
<m>\begin{align}
\nabla_{W^{(l)}} J(W,b;x,y) &amp;= \delta^{(l+1)} (a^{(l)})^T, \\
\nabla_{b^{(l)}} J(W,b;x,y) &amp;= \delta^{(l+1)}.
\end{align}</m>



__Implementation note:__ In steps 2 and 3 above, we need to compute <m>\textstyle f'(z^{(l)}_i)</m> for each value of <m>\textstyle i</m>. Assuming <m>\textstyle f(z)</m> is the sigmoid activation function, we would already have <m>\textstyle a^{(l)}_i</m> stored away from the forward pass through the network.  Thus, using the expression that we worked out earlier for <m>\textstyle f'(z)</m>, 
we can compute this as <m>\textstyle f'(z^{(l)}_i) = a^{(l)}_i (1- a^{(l)}_i)</m>.   

Finally, we are ready to describe the full gradient descent algorithm.  In the pseudo-code
below, <m>\textstyle \Delta W^{(l)}</m> is a matrix (of the same dimension as <m>\textstyle W^{(l)}</m>), and <m>\textstyle \Delta b^{(l)}</m> is a vector (of the same dimension as <m>\textstyle b^{(l)}</m>). Note that in this notation, 
"<m>\textstyle \Delta W^{(l)}</m>" is a matrix, and in particular it isn't "<m>\textstyle \Delta</m> times <m>\textstyle W^{(l)}</m>." We implement one iteration of batch gradient descent as follows:


1. Set <m>\textstyle \Delta W^{(l)} := 0</m>, <m>\textstyle \Delta b^{(l)} := 0</m> (matrix/vector of zeros) for all <m>\textstyle l</m>.
2. For <m>\textstyle i = 1</m> to <m>\textstyle m</m>, 

    1. Use backpropagation to compute <m>\textstyle \nabla_{W^{(l)}} J(W,b;x,y)</m> and <m>\textstyle \nabla_{b^{(l)}} J(W,b;x,y)</m>.
    2. Set <m>\textstyle \Delta W^{(l)} := \Delta W^{(l)} + \nabla_{W^{(l)}} J(W,b;x,y)</m>. 
    3. Set <m>\textstyle \Delta b^{(l)} := \Delta b^{(l)} + \nabla_{b^{(l)}} J(W,b;x,y)</m>. 

3. Update the parameters:
<m>\begin{align}
W^{(l)} &amp;= W^{(l)} - \alpha \left[ \left(\frac{1}{m} \Delta W^{(l)} \right) + \lambda W^{(l)}\right] \\
b^{(l)} &amp;= b^{(l)} - \alpha \left[\frac{1}{m} \Delta b^{(l)}\right]
\end{align}</m>


To train our neural network, we can now repeatedly take steps of gradient descent to reduce our cost function <m>\textstyle J(W,b)</m>.
