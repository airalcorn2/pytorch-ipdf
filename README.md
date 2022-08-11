# PyTorch Implicit-PDF

This repository contains a minimal PyTorch implementation of the implicit-PDF (IPDF) model described in "[Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold"](https://arxiv.org/abs/2106.05965).
[`train_ipdf.py`](train_ipdf.py) trains an IPDF model using the same hyperparameters described in the paper (except for a smaller batch size) on the SYMSOL I dataset introduced by the authors.
Note, the training script isn't an attempt to exactly reproduce the paper's results.
I use the full training set during training, and I use 4,095 random rotations when evaluating on the test set (as opposed to the 2,000,000 gridded rotations).
With that being said, after three epochs, the model achieves an average log-likelihood of 3.67 on the test set, which is less than the 4.10 from Table 1, but already much better than the average log-likelihoods for the other methods in Table 1, i.e., the port seems to be working as expected.
