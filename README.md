# What is this repository?

This repository contains code for paper "[On Minimax Optimality of GANs for Robust Mean Estimation][paper]".

[paper]: https://cs.uwaterloo.ca/~k77wu

We implemented f-GAN, MMD-GAN (with Gaussian kernel) and Wasserstein GAN (with Euclidean norm as ground cost). These models are tested under Huber's contamination model.

## Usage
To install dependency, run
```
pip install -r requirements.txt
```

Run the following scripts containing detailed parameter configurations:
```
bash test_fgan.sh
bash test_mmd.sh
bash test_sinkhorn.sh
```

