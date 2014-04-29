---
layout: post
title:  "RICA"
date:   2013-08-28 13:32:12
categories: unsupervised
---
In this exercise, you will implement a one-layer RICA network and apply them to MNIST images. 

You will build on MATLAB starter code which we have provided in the [starter code](https://github.com/amaas/stanford_dl_ex). You need only write code at places indicated by `YOUR CODE HERE`. You will modify the files `softICACost.m` and `zca2.m`

### Step 0: Prerequisites ###
#### Step 0a: Read runSoftICA.m ####
The file `runSoftICA.m` is the "main" script. It handles loading data, preprocessing it, and calling `minFunc` with the appropriate parameters. Be sure to understand how this file works before moving further. 

### Step 0b: Implement zca2.m ###
Implement the ZCA transform in `zca2.m`. You should be able to copy and paste your code from [Exercise: PCA Whitening]({{site.baseurl}}/unsupervised/ExercisePCAWhitening) if you have successfully completed that exercise.

### Step 1: RICA cost and gradient ###

First, let us derive the gradient of the RICA reconstruction cost using the backpropagation idea.

#### Step 1a: Deriving gradient using Backpropagation ####

Recall the [RICA]({{site.baseurl}}/unsupervised/RICA) reconstruction cost term:

<m>\lVert W^TWx - x \rVert_2^2</m>

where <m>W</m> is the weight matrix and <m>x</m> is the input.

We would like to find <m>\nabla_W \lVert W^TWx - x \rVert_2^2</m> - the derivative of the term with respect to the '''weight matrix''', rather than the '''input''' as in the earlier two examples. We will still proceed similarly though, seeing this term as an instantiation of a neural network:

<img src="{{site.baseurl}}/images/Backpropagation_Method_Example_3.png" width="100%">

The weights and activation functions of this network are as follows:

<table align="center">
<tr>
<td width="80px"><m>\text{Layer}</m></td>
<td width="80px"><m>\text{Weight}</m></td>
<td width="150px"><m>\text{Activation function}</m></td></tr>
<tr>
<td><m>1</m></td>
<td><m>W</m></td>
<td><m>f(z_i) = z_i</m></td>
</tr>
<tr>
<td><m>2</m></td>
<td><m>W^T</m></td>
<td><m>f(z_i) = z_i</m></td>
</tr>
<tr>
<td><m>3</m></td>
<td><m>I</m></td>
<td><m>f(z_i) = z_i - x_i</m></td>
</tr>
<tr>
<td><m>4</m></td>
<td><m>\text{N/A}</m></td>
<td><m>f(z_i) = z_i^2</m></td>
</tr>
</table>

To have <m>J(z^{(4)}) = F(x)</m>, we can set <m>J(z^{(4)}) = \sum_k J(z^{(4)}_k)</m>.

Now that we can see <m>F</m> as a neural network, we can try to compute the gradient <m>\nabla_W F</m>. However, we now face the difficulty that <m>W</m> appears twice in the network. Fortunately, it turns out that if <m>W</m> appears multiple times in the network, the gradient with respect to <m>W</m> is simply the sum of gradients for each instance of <m>W</m> in the network (you may wish to work out a formal proof of this fact to convince yourself). With this in mind, we will proceed to work out the deltas first:

<table align="center">
<tr>
<th width="80px"><m>\text{Layer}</m></th>
<th width="150px"><m>\text{Derivative of activation function }f'</m></th>
<th width="150px"><m>\text{Delta}</m></th>
<th width="150px"><m>\text{Input }z \text{ to this layer}</m></th>
</tr>
<tr>
<td><m>4</m></td>
<td><m>f'(z_i) = 2z_i</m></td>
<td><m>f'(z_i) = 2z_i</m></td>
<td><m>(W^TWx - x)</m></td>
</tr>
<tr>
<td><m>3</m></td>
<td><m>f'(z_i) = 1</m></td>
<td><m>\left( I^T \delta^{(4)} \right) \bullet 1</m></td>
<td><m>W^TWx</m></td>
</tr>
<tr>
<td><m>2</m></td>
<td><m>f'(z_i) = 1</m></td>
<td><m>\left( (W^T)^T \delta^{(3)} \right) \bullet 1</m></td>
<td><m>Wx</m></td>
</tr>
<tr>
<td><m>1</m></td>
<td><m>f'(z_i) = 1</m></td>
<td><m>\left( W^T \delta^{(2)} \right) \bullet 1</m></td>
<td><m>x</m></td>
</tr>
</table>

To find the gradients with respect to <m>W</m>, first we find the gradients with respect to each instance of <m>W</m> in the network.

With respect to <m>W^T</m>:

<m>
\begin{align}
\nabla_{W^T} F &amp; = \delta^{(3)} a^{(2)T} \\
&amp; = 2(W^TWx - x) (Wx)^T
\end{align}
</m>

With respect to <m>W</m>:

<m>
\begin{align}
\nabla_{W} F &amp; = \delta^{(2)} a^{(1)T} \\
&amp; = (W)(2(W^TWx -x)) x^T
\end{align}
</m>

Taking sums, noting that we need to transpose the gradient with respect to <m>W^T</m> to get the gradient with respect to <m>W</m>, yields the final gradient with respect to <m>W</m> (pardon the slight abuse of notation here):

<m>
\begin{align}
\nabla_{W} F &amp; = \nabla_{W} F + (\nabla_{W^T} F)^T \\
&amp; = (W)(2(W^TWx -x)) x^T + 2(Wx)(W^TWx - x)^T
\end{align}
</m>

#### Step 1b: Implement cost and gradient ####
In the file `softICACost.m`, implement the RICA cost and gradient. The cost we use is: 

<m>
\min_{W} \quad \lambda \left\|Wx\right\|_1  + \frac{1}{2} \left\| W^T Wx - x \right\|_2^2
</m>

Note that this is slightly different than the cost used in the gradient derivation section above (because we have added the L1 regularization and scaled the reconstruction term down by 0.5). To implement the L1-norm, we suggest using: <m> f(x) = \sqrt{x^2 + \epsilon} </m> for some small <m>\epsilon</m>. In this exercise, we find <m>\epsilon=0.01</m> to work well. 

When done, check your gradient implementation. You could do this either using your own `checkNumericalGradient.m` from previous sections, or by using minFunc's built-in checker.

#### Comparison Results ####
`TODO`
