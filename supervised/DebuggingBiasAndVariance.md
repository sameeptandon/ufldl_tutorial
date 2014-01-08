---
layout: post
title:  "Debugging: Bias and Variance"
date:   2013-08-28 13:32:12
categories: authoring, development
---

Thus far, we have seen how to implement several types of machine learning algorithms.  Our usual goal is to achieve the highest possible prediction accuracy on novel test data that our algorithm did not see during training.  It turns out that the our accuracy on the <i>training</i> data is an upper bound on the accuracy we can expect to achieve on the testing data.  (We can sometimes get lucky and do better on a small sample of test data;  but on average we will tend to do worse.)  In some sense, the training data is "easier" because the algorithm has been trained for those examples specifically and thus there is a gap between the training and testing accuracy.

