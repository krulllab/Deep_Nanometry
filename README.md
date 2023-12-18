# Deep Nanometry

Yuichiro Iwamoto<sup>1</sup><sup>†</sup>, Benjamin Salmon<sup>2</sup><sup>†</sup>, Yoshiyuki Tsuyama<sup>3</sup>, Miu Tamamitsu<sup>1</sup>, Yusuke Yoshioka<sup>3</sup>, Alexander Krull<sup>2</sup><sup>\*</sup> and Sadao Ota<sup>1</sup><sup>\*</sup></br>

<sup>1</sup>Research Center for Advanced Science and Technology, The University of Tokyo, Meguro-ku 4-6-1, Shibuya, Tokyo 153-8904, Japan.<br>
<sup>2</sup>School of Computer Science, University of Birmingham, B15 2TT, Birmingham, United Kingdom.<br>
<sup>3</sup>Department of Molecular and Cellular Medicine, Institute of Medical Science, Tokyo Medical University, Nishishinjuku 6-7-1, Shinjuku, Tokyo 160-0023, Japan.<br>

The introduction of unsupervised methods in denoising has shown that unpaired noisy data can be used to train denoising networks, which can not only produce high quality results but also enable us to sample multiple possible diverse denoising solutions. 
However, these systems rely on a probabilistic description of the imaging noise--a noise model.
Until now, imaging noise has been modelled as pixel-independent in this context.
While such models often capture shot noise and readout noise very well, they are unable to describe many of the complex patterns that occur in real life applications.
Here, we introduce a novel learning-based autoregressive noise model to describe imaging noise and show how it can enable unsupervised denoising for settings with complex structured noise patterns.
We explore different ways to train a model for real life imaging noise and show that our deep autoregressive noise model has the potential to greatly improve denoising quality in structured noise datasets.
We showcase the capability of our approach on various simulated datasets and on real photo-acoustic imaging data.

### Information

Code for the publication Deep Nanometry: An optofluidic high-throughput nanoparticle analyzer with enhanced sensitivity via unsupervised deep learning-based denoising.

### Dependencies
We recommend installing the dependencies in a conda environment. If you haven't already, install miniconda on your system by following this [link](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).<br>
Once conda is installed, create and activate an environment by entering these lines into a command line interface:<br>
1. `conda create --name dnm`
2. `conda activate dnm`


Next, install PyTorch and torchvision for your system by following this [link](https://pytorch.org/get-started/locally/).<br> 
After that, you're ready to install the dependencies for this repository:<br>
`pip install lightning jupyterlab matplotlib tifffile scikit-image tensorboard`

### Getting Started
The 'examples' directory contains notebooks for denoising and carrying out the analyses in the paper. They assume data has been stored as .npy files in a 'data' directory as numpy ndarrays with dimensions [Number, Channels, Width]. 
