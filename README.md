# Deep Learning Labs

A collection of hands-on deep learning notebooks covering the fundamentals of neural networks — from scratch implementations to modern frameworks, spanning computer vision, NLP, and generative sequence modeling.

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

---

## Notebooks Overview

### 1. MLP from Scratch — Pure NumPy
**`mlp_classification_python.ipynb`**

Binary classification on a non-linearly separable dataset, implemented entirely with NumPy — no frameworks.

- Manual forward pass (ReLU + Sigmoid activations)
- Hand-coded backpropagation using the chain rule
- Two optimizers compared: **gradient descent** vs. **momentum**
- ~90% accuracy on `make_circles` dataset

> **Skills demonstrated:** mathematical foundations of deep learning, gradient computation from first principles, optimization techniques.

---

### 2. MLP Binary Classification — PyTorch
**`mlp_binary_classification_pytorch.ipynb`**

Three progressively abstract PyTorch implementations of the same MLP, illustrating the full API spectrum.

| Approach | Description |
|---|---|
| Manual | Raw tensor ops + `loss.backward()` |
| Sequential | `nn.Sequential` declarative model |
| Module | Custom `nn.Module` subclass |

- Dataset: `make_circles` (1,000 samples, noise=0.2)
- Optimizer: SGD with/without momentum
- 10,000 training epochs → loss converges from ~0.69 to ~0.24

> **Skills demonstrated:** PyTorch autograd, multiple levels of API abstraction, training loop design.

---

### 3. CNN Image Classification — PyTorch (Part 1)
**`cnn_pytorch_part_1.ipynb`**

Convolutional networks for image recognition on two standard benchmarks.

| Dataset | Architecture | Accuracy |
|---|---|---|
| MNIST | Conv → Pool → FC | **94.3%** (10 epochs) |
| CIFAR-10 | Conv → Pool → FC | 43.6% (25 epochs) |

- 3×3 convolutional kernels, 32 filters, max pooling
- `DataLoader` for mini-batch training
- Visualization of learned convolutional filters

> **Skills demonstrated:** CNN architecture design, PyTorch `DataLoader`, evaluation on train/test splits, filter visualization.

---

### 4. CNN Interpretability — Deep Dream & Adversarial Examples (Part 2)
**`cnn_pytorch_part_2.ipynb`**

Gradient-based input optimization using a pre-trained VGG19 network to understand what CNNs actually learn.

- **Deep Dream:** gradient ascent to amplify feature activations → psychedelic image transformations
- **Adversarial examples:** targeted gradient ascent to fool the classifier (cat → "reflex camera" in 40 iterations)
- Gradient normalization to stabilize optimization
- ImageNet preprocessing pipeline for VGG19

> **Skills demonstrated:** pre-trained model usage, gradient w.r.t. inputs, adversarial ML, model interpretability techniques.

---

### 5. RNN Sentiment Analysis — Keras
**`rnn_keras_introduction.ipynb`**

Sentiment analysis (many-to-one) on the IMDB movie reviews dataset using two architectures.

| Model | Approach | Test Accuracy |
|---|---|---|
| Embedding + Mean | Bag-of-words with learned embeddings | **83.7%** |
| Embedding + LSTM | Sequential modeling (100 units) | **84.9%** |

- Word embedding layer (5,000 vocab → 32 dimensions)
- `pad_sequences` for variable-length input handling
- Embedding space exploration via Euclidean distance between learned word vectors

> **Skills demonstrated:** NLP preprocessing, word embeddings, LSTM, Keras Functional and Sequential APIs.

---

### 6. RNN Music Generation — Sequence Modeling (Part 2)
**`rnn_keras_part_2.ipynb`**

One-to-many language model trained on 36 MIDI files of Bach's Cello Suites to generate original musical compositions.

- MIDI → pitch sequence conversion via `pretty_midi`
- Stacked LSTM architecture (32 → 32 → 256 units) with dropout (0.3)
- Softmax over 79 musical notes
- **Temperature-scaled sampling** to control creativity vs. coherence
- Generated sequences exported as playable MIDI files

> **Skills demonstrated:** sequence-to-sequence data preparation, stacked LSTMs, generative modeling, dropout regularization, early stopping.

---

## Learning Progression

```
Fundamentals           Frameworks              Advanced Topics
─────────────────      ─────────────────       ─────────────────
NumPy MLP         →    PyTorch MLP        →    CNN Interpretability
(from scratch)         (3 API levels)          (Deep Dream, Adversarial)

                        PyTorch CNN        →    RNN Sequence Generation
                        (MNIST, CIFAR-10)       (Music, Language Models)
```

The notebooks are ordered by increasing complexity:
1. Mathematical foundations (no framework)
2. Framework abstractions (PyTorch, Keras)
3. Applied topics (CV, NLP, generative models)

---

## Key Competencies

| Area | Demonstrated in |
|---|---|
| Neural network math (backprop, chain rule) | `mlp_classification_python` |
| PyTorch (autograd, training loops, DataLoader) | `mlp_binary_classification_pytorch`, `cnn_pytorch_part_1` |
| TensorFlow / Keras | `rnn_keras_introduction`, `rnn_keras_part_2` |
| Computer Vision (CNN, filters, VGG19) | `cnn_pytorch_part_1`, `cnn_pytorch_part_2` |
| NLP (embeddings, LSTM, sentiment analysis) | `rnn_keras_introduction` |
| Generative modeling (language model, sampling) | `rnn_keras_part_2` |
| Model interpretability (Deep Dream, adversarial) | `cnn_pytorch_part_2` |
| Optimization (SGD, momentum, Adam) | multiple notebooks |

---

## Getting Started

```bash
git clone https://github.com/Matthis44/deep-learning-labs.git
cd deep-learning-labs
pip install -r requirements.txt
jupyter notebook
```

The notebooks are saved with their outputs, so they can be read on GitHub without running them.
`rnn_keras_part_1.ipynb` is an alternative solution of the same IMDB lab as `rnn_keras_introduction.ipynb`.

---

## Context

These notebooks are lab sessions from the Deep Learning course at Télécom Paris
(lab statements by Geoffroy Peeters). The code that completes the exercises, the experiments
and the analyses are mine.
