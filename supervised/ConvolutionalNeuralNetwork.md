---
layout: post
title:  "Convolutional Neural Network"
date:   2013-08-28 13:32:12
categories: supervised
---

### Overview ###

A Convolutional Neural Network (CNN) is comprised of one or more convolutional layers (often with a subsampling step) and then followed by one or more fully connected layers as in a standard [multilayer neural network]({{site.baseurl}}/supervised/MultiLayerNeuralNetworks).  The architecture of a CNN is designed to take advantage of the 2D structure of an input image (or other 2D input such as a speech signal).  This is achieved with local connections and tied weights followed by some form of pooling which results in translation invariant features.  Another benefit of CNNs is that they are easier to train and have many fewer parameters than fully connected networks with the same number of hidden units.   In this article we will discuss the architecture of a CNN and the back propagation algorithm to compute the gradient with respect to the parameters of the model in order to use gradient based optimization.  See the respective tutorials on [convolution]({{site.baseurl}}/supervised/FeatureExtractionUsingConvolution) and [pooling]({{site.baseurl}}/supervised/Pooling) for more details on those specific operations.

###Architecture###

A CNN consists of a number of convolutional and subsampling layers optionally followed by fully connected layers.  The input to a convolutional layer is a <m>m \text{ x } m \text{ x } r</m> image where <m>m</m> is the height and width of the image and <m>r</m> is the number of channels, e.g. an RGB image has <m>r=3</m>.   The convolutional layer will have <m>k</m> filters (or kernels) of size <m>n \text{ x } n \text{ x } q</m> where <m>n</m> is smaller than the dimension of the image and <m>q</m> can either be the same as the number of channels <m>r</m> or smaller and may vary for each kernel.  The size of the filters gives rise to the locally connected structure which are each convolved with the image to produce <m>k</m> feature maps of size <m>m-n+1</m>.  Each map is then subsampled typically with mean or max pooling over <m>p \text{ x } p</m> contiguous regions where p ranges between 2 for small images (e.g. MNIST) and is usually not more than 5 for larger inputs.  Either before or after the subsampling layer an additive bias and sigmoidal nonlinearity is applied to each feature map.  The figure below illustrates a full layer in a CNN consisting of convolutional and subsampling sublayers.  Units of the same color have tied weights.

<center>
<img src="{{site.baseurl}}/images/Cnn_layer.png">
</center>
<center>
<p style='width:600px'>Fig 1: First layer of a convolutional neural network with pooling. Units of the same color have tied weights and units of different color represent different filter maps.</p>
</center>

After the convolutional layers there may be any number of fully connected layers.  The densely connected layers are identical to the layers in a standard [multilayer neural network]({{site.baseurl}}/supervised/MultiLayerNeuralNetworks).

### Back Propagation ###

Let <m>\delta^{(l+1)}</m> be the error term for the <m>(l+1)</m>-st layer in the network with a cost function <m>J(W,b ; x,y)</m> where <m>(W, b)</m> are the parameters and <m>(x,y)</m> are the training data and label pairs.  If the <m>l</m>-th layer is densely connected to the <m>(l+1)</m>-st layer, then the error for the <m>l</m>-th layer is computed as

<m>
   \begin{align}
   \delta^{(l)} = \left((W^{(l)})^T \delta^{(l+1)}\right) \bullet f'(z^{(l)})
   \end{align}
</m>

and the gradients are

<m>
\begin{align}
   \nabla_{W^{(l)}} J(W,b;x,y) &amp;= \delta^{(l+1)} (a^{(l)})^T, \\
   \nabla_{b^{(l)}} J(W,b;x,y) &amp;= \delta^{(l+1)}.
\end{align}
</m>

If the <m>l</m>-th layer is a convolutional and subsampling layer then the error is propagated through as

<m>
   \begin{align}
   \delta_k^{(l)} = \text{upsample}\left((W_k^{(l)})^T \delta_k^{(l+1)}\right) \bullet f'(z_k^{(l)})
   \end{align}
</m>

Where <m>k</m> indexes the filter number and <m>f'(z_k^{(l)})</m> is the derivative of the activation function. The <code>upsample</code> operation has to propagate the error through the pooling layer by calculating the error w.r.t to each unit incoming to the pooling layer. For example, if we have mean pooling then <code>upsample</code> simply uniformly distributes the error for a single pooling unit among the units which feed into it in the previous layer.  In max pooling the unit which was chosen as the max receives all the error since very small changes in input would perturb the result only through that unit.

Finally, to calculate the gradient w.r.t to the filter maps, we rely on the border handling convolution operation again and flip the error matrix <m>\delta_k^{(l)}</m> the same way we flip the filters in the [convolutional layer]({{site.baseurl}}/supervised/FeatureExtractionUsingConvolution).

<m>
   \begin{align}
     \nabla_{W_k^{(l)}} J(W,b;x,y) &amp;= \sum_{i=1}^m (a_i^{(l)}) \ast \text{rot90}(\delta_k^{(l+1)},2), \\
     \nabla_{b_k^{(l)}} J(W,b;x,y) &amp;=  \sum_{a,b} (\delta_k^{(l+1)})_{a,b}.
   \end{align}
</m>

Where <m>a^{(l)}</m> is the input to the <m>l</m>-th layer, and <m>a^{(1)}</m> is the input image. The operation <m>(a_i^{(l)}) \ast \delta_k^{(l+1)}</m> is the "valid" convolution between <m>i</m>-th input in the <m>l</m>-th layer and the error w.r.t. the <m>k</m>-th filter.
