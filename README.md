# Bayesian RG Flow in Neural Network Field Theories

# Abstract
The Neural Network Field Theory correspondence (NNFT) is a mapping from neural network (NN)
architectures into the space of statistical field theories (SFTs). The Bayesian renormalization group
(BRG) is an information-theoretic coarse graining scheme that generalizes the principles of the Exact
Renormalization Group (ERG) to arbitrarily parameterized probability distributions, including those of
NNs. In BRG, coarse graining is performed in parameter space with respect to an information-theoretic
distinguishability scale set by the Fisher information metric. In this paper, we unify NNFT and BRG
to form a powerful new framework for exploring the space of NNs and SFTs, which we coin BRG-NNFT. 
With BRG-NNFT, NN training dynamics can be interpreted as inducing a flow in the space of SFTs from the 
information-theoretic ‘IR’ → ‘UV’. Conversely, applying an information-shell coarse graining to the 
trained network’s parameters induces a flow in the space of SFTs from the information-theoretic 
‘UV’ → ‘IR’. When the information-theoretic cutoff scale coincides with a standard momentum scale, 
BRG is equivalent to ERG. We demonstrate the BRG-NNFT correspondence on two analytically tractable 
examples. First, we construct BRG flows for trained, infinite-width NNs, of arbitrary depth, with 
generic activation functions. As a special case, we then restrict to architectures with a single
infinitely-wide layer, scalar outputs, and generalized cos-net activations. In this case, we show that
BRG coarse-graining corresponds exactly to the momentum-shell ERG flow of a free scalar SFT in
Euclidean space. Our analytic results are corroborated by a numerical experiment in which an ensemble
of asymptotically wide NNs are trained and subsequently renormalized using an information-shell BRG
scheme.

# BibTeX Citation
If you use our code, please cite the paper
```
Insert citation here
```
