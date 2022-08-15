# PyTorch implicit-PDF

This repository contains a minimal PyTorch implementation of the implicit-PDF (IPDF) model described in "[Implicit-PDF: Non-Parametric Representation of Probability Distributions on the Rotation Manifold"](https://arxiv.org/abs/2106.05965).
[`train.py`](train.py) trains an IPDF model using the same hyperparameters described in the paper on the SYMSOL I dataset introduced by the authors.
[`evaluate.py`](evaluate.py) evaluates the trained model on the SYMSOL I test set using the rotations grid described in the paper.