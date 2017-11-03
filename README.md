# MAML

## Intro

This repository implements the paper, [Model-Agnostic Meta-Leanring for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400).

## Requirements

- Tensorflow (v1.3 or higher)
- better_exceptions, tqdm, Pillow, etc.

## Training statisitics

- [x] Sinusoid

  ![Sinusoide Result](/assets/sinusoid_result.png)
  Red line - ground truth, Red dots - given observation, Blue line - predicted line after 1 sgd step

- [x] Omniglot
  - [x] Omniglot Testing (multiple descent steps)

  ![Ominglot Result](/assets/omniglot-5way-1shot.png)
  - valid_acc_{0,1} means accuracy after 1 and 2 SGD steps. The valid_acc is the accuracy after the weights are trained with 3 SGD steps.

## Training

### Download datasets

Downlaod [Omniglot dataset](https://github.com/brendenlake/omniglot/tree/master/python) from the link. Only `images_background.zip` and `images_evalueation.zip` are required.

Unzip on the directory `(repository)/datasets/omniglot/`, so the directory shoud looks like `(repo)/datasets/omniglot/{images_background,images_evaluation}`.

### Run train

- Run sinusoide: `python sinusoide.py`
- Run omniglot: `python omniglot.py`

Change the hyperparameters accordingly as you want. Please check at the bottom of each script.

## TODO
- [ ] Mini Imagenet Training
- [ ] Robotic Simulation.

## Acknowledgement

- Author's original implementation: [link](https://github.com/cbfinn/maml)
