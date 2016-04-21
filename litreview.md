# Review of Visualizing Neural Networks

Below is a summary of existing efforts to visualize neural networks. This review is part of my research into neural networks to visualize them as my final project as a candidate for a Master's in Data Science.

****
## Overview and Motivations

There are two objectives in visualizing neural networks: to educate and to understand.

Though neural networks have been around for a while, there is little understanding of why and how neural networks work. This has motivated a lot of academic work in how to visualize neural networks to understand what's going on at different levels of architecture and improve performance.

The audience for my master's project is a general public with no background in computer science or machine learning. While the work of deep learners brings one perspective to this goal, it is not the same audience with the same goals. Machine learning practitioners visualize networks to understand how they work in order to improve performance (and in theory improve theory).

The motivation in visualizing neural networks for a general public is to educate. More concretely, my objectives are to provide transparency and satisfy curiousities. Neural networks are being used in many every day applications and those interested should have an opportunity to understand how they work.

Below I will describe some of the academic approaches. They are interesting and enlightening but also the set of alternative attempts is limited.

****
## Academic work

#### Zeiler and Fergus, Deconvnet, 2014

A common question is what does a neural network learn to be important concepts for image classification at intermediate layers. Visualization of a convolutional network using a deconvolutional network (deconvnet) leverages the ability of a deconvnet to map feature activities back to the original pixel input space. Deconvnets invert - as much as possible - the actions of a neural network.  Zeiler and Fergus explored the method of understanding what concepts does a neural network learn founded in their work deconvolutional networks ([Deconvnets][Zeiler and Fergus]) in 2011. Proposed as a way to do unsupervised learning, deconvnets reveal what concepts of image classes are developed by a convolutional network. I will omit the details of the inversion here as it is discussed in detail in their paper.

As part of the same goal to understand why neural net works are working, they perform sensitivity analysis by occluding many small sections of an image and analysing classifier output. This helps identifiy what parts of the image the classifier finds important in correctly classifying the image.

Zeiler and Fergus produced visualizations using the popularized convnet architectures developed by [LeCun et. al.][Lecun] and [Krishevsky et. al.][Krishevsky]. The visual analysis helped them beat the AlexNet 2012 single-model result by 1.7% using smaller strides (2 vs. 4) and smaller filters (7x7 vs. 11x11).

#### Jason Yosinski, DeepVis, 2015

Jason Yosinski, DeepVis, 2015 published a [paper][Yosinski Paper] and open source tool for visualizing a neural network in real time, with an image or camera feed. The tool is like a dashboard (my term) of the network, enabling the user to visualize and interact with intermediary results of the network.

The strength of DeepVis may be in its simplicity: including the visualization of the most straight-forward approach to plot activations values enables the viewer to see all the data:

> Although this visualization is simple to implement, we find it informative because all data flowing through the network can be visualized. There is nothing mysterious happening behind the scenes. (pg 4)

Though the approaches are straight-forward, these visualizations yielded new and surprising intuitions:

* Layers demonstrate locality, that is become detectors of different real-world objects like flowers or faces. This suggests that intermediate layers become responsible for different concepts important in a well-trained classifer.
* When an image does not include anything from the training set of classes, the real-time probability vector exposed a high sensitivity to small changes in input. In other words, shifting in your chair could mean instead of being classified as a cat you are a lamp.
* Although higher layers are sensitive, lower level computations are robust: "....the network learns to identify these concepts simply because they represent useful partial information for making a later classification decision"

Yosinski and Zeigler both expressed surprise at finding the features learned by intermediate layers are discernable and important to final classification. This is interesting because it should that the network is in effect, in that learning the concept of text in an image has become an important way for the network to define a linear boundrary in hidden layers.

> That said, not all features correspond to natural parts, raising the possibility of a different decomposition of the world than humans might expect. These visualizations suggest that further study into the exact nature of learned representations — whether they are local to a single channel or distributed across several — is likely to be interesting. (pg 9)

The features of the tool and links on how to install it are available here: [DeepVis][Yosinski DeepVis].

****
## Related Work

#### Andrej Karpathy, etc.

* [Visualizing what ConvNets learn][Visualizing what ConvNets learn]: An list of useful ways to visualize a variety components of neural networks, including layer activations and visualizing high dimensional feature topologies.


#### Daniel Smilkov and Shan Carter, A Neural Network Playground, 2016

[Neural Network Playground][Neural Network Playground] developed by Daniel Smilkov and Shan Carter offers a user-friendly and attractive interactive tool for understanding at a high level how neural networks work and how you might tune them.


#### Colah, etc.

**[Neural Networks, Manifolds, and Topology][Neural Networks, Manifolds, and Topology]**

Colah and Karpathy both provide helpful animations of how space can be warped such that non-linear data is linearly seperable. One thing I like about Colah's article in particular is he breaks this down into it's component parts (weighting, translation and activation function) and provides a visualization of that process in fine-grained detail.

Colah's intuitive explanations and visualizations of the challenge of warping space such that naturally non-linear data may be linearly seperated. I highly recommend a reading to gain an intuition through simple examples. Colah builds on the intuitions gleaned from this simplified examples to theorize about lower-bounds on the dimensionality requirements of neural networks (i.e. the Manifold Hypothesis).

Colah also has some very good articles on visualizing high-dimensional data, which is relevant but perhaps tangential to this topic:

* [Visualizing Representations: Deep Learning and Human Beings](http://colah.github.io/posts/2015-01-Visualizing-Representations/)
* [Visualizing MNIST: An Exploration of Dimensionality Reduction](http://colah.github.io/posts/2014-10-Visualizing-MNIST/)

#### [Inceptionism: Going Deeper into Neural Networks][Inceptionism]

Google also "inverts" a trained neural network classifier to identify errors in training. If you have a classifier trained on 10 images and you want to determine what types of image patterns the network is learned, you can start with an image of random noise and tweak the image until you get something the classifier finds to be a banana with high probability.

****
## Conclusions

* Approaches by academics have been targeted at research interests, and while they may not be directly instructive in creating interactions for a general public, they still may offer insight into best approaches.
* There have been few attempts at including interactivity and none at gamification as a way to tackle the education objective, with the exception of the Neural Network playground.
* Many of the references including so far are in the domain of image classification, so it will be a thought adventure in how to generalize learnings and approaches into other domains and architecutures.

[Yosinski Paper]: http://yosinski.com/media/papers/Yosinski__2015__ICML_DL__Understanding_Neural_Networks_Through_Deep_Visualization__.pdf
[Yosinski DeepVis]: http://yosinski.com/deepvis
[Deconvnet]: https://www.cs.nyu.edu/~gwtaylor/publications/zeilertaylorfergus_iccv2011.pdf
[Neural Network Playground]: http://playground.tensorflow.org
[Lecun]: http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf
[Krishevsky]: http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
[Visualizing what ConvNets learn]: http://cs231n.github.io/understanding-cnn/
[Neural Networks, Manifolds, and Topology]: http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
[Inceptionism]: http://googleresearch.blogspot.com.es/2015/06/inceptionism-going-deeper-into-neural.html

