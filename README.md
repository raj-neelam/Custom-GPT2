# GPT-2 Implementation

This repository contains an implementation of the GPT-2 model architecture. You can use this to train a model from scratch on your own custom data, or you can load the original pre-trained weights from OpenAI's GPT-2.

## Model Overview

| Transformer Architecture | Model Architecture Diagram |
| :----------------------- | :------------------------- |
| **Generative Pre-trained Transformer (GPT-2)**<br><br>The model is built using a decoder-only Transformer architecture.<br><br>**Key Features:**<br><ul><li>**Token & Positional Embeddings:** Converts input tokens into continuous vectors and adds position information.</li><li>**Transformer Blocks:** A sequence of identical layers, each containing:<ul><li>*Masked Multi-Head Self-Attention:* Allows the model to selectively focus on relevant preceding tokens while preventing attention to future tokens.</li><li>*Layer Normalization:* Applied before the attention and feed-forward networks for training stability.</li><li>*Feed-Forward Network:* A Multi-Layer Perceptron applied to each token independently.</li></ul></li><li>**Output Head:** Projects the final hidden states back to the vocabulary size to predict the next token probabilities.</li></ul> | <img src="model_architecture.png" alt="Model Architecture" style="max-height: 800px; width: auto;" /> |

<br>

### Specifications & Capabilities

- **Train on Your Own Data:** The codebase allows you to easily plug in your custom text datasets to train the transformer from scratch.
- **Load Pretrained Weights:** You can load the official pretrained weights from OpenAI's GPT-2 directly into this model architecture for inference and fine-tuning.
