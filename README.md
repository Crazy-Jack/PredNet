# uPNC2021-aniekan

Summer project to understand internal representations of PredNet.

### Research Description:

PredNet is a Machine Learning (ML) model for next-frame prediction inspired by Predictive Coding.  It has been demonstrated that PredNet can replicate phenomena from the primate visual cortex, and it shows promise as a neuroscientific tool to understand the brain.  The goal of my summer research project was to investigate the inner workings of PredNet, specifically to understand the features to which it is tuned.  To this end, I employed feature visualization: an ML technique typically used to probe the preferred/anti-preferred stimulus features for units in Convolutional Neural Networks (CNNs).  This method optimizes the image to maximize/minimize the activation of the desired unit (ranging from single neurons to whole layers).  However, unlike CNNs, the internal states of PredNet are dynamic.  Thus, we require a generalized notion of extremal activity.  My project focused on the following extremes: Strong Sustained Response (SSR), Weak Sustained Response (WSR), and Strong Impulse Response (SIR).  Features that elicit an SSR/WSR are optimized to cause the neuron to have maximal/minimal activity throughout the entire sequence.  Features that elicit a SIR are optimized to cause the neuron to have maximal activity at a single time frame (the last frame in this study).  These responses were explored for neurons in the top-most layer within the recurrent representation module (R) and the feed-forward target modules (A).  The feature visualization demonstrated functional differences between R and A units.  Firstly the SSR feature for A units displayed jittering motion while the WSR feature lacked motion.  This result suggests that A units are sensitive to motion.  Secondly, the SIR features for the R units span the entire sequence, while SIR features for the A units only appear during the last few frames of the sequences.  This finding supports the notion that the R units do encode long-term dependencies, which is essential for building an internal model of the world.

The full results are displayed on my [uPNC Poster](https://docs.google.com/presentation/d/1swV622Tti9ab4fdUZ7L87N5N37Wzt01zPbiCxjuins8/edit?usp=sharing).

### Directories

ConvLSTM_MNST: Getting familiar with ConvLSTM.  Attempted to train on MNIST but realized the paper
	       did not use a simple ConvLSTM but an autoencoder-ConvLSTM.
	       
img_optim: Attempt at visualizing the feature of prednet units

kitti_tuning: Determining a unit's preffered/anti-preffered samples from the KITTI dataset

optical_flow_tuning: testing prednet on synethic, optical-flow stimuli to explore possible direction tunning. (INCOMPLETE)

prednet_pytorch: Re-implementation of [prednet](https://arxiv.org/abs/1906.11902v2)
