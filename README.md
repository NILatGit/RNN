# RNN
# RNN Text Generation with PyTorch

Welcome to the RNN Text Generation project! This repository provides a PyTorch-based implementation of a Recurrent Neural Network (RNN) for generating text character-by-character. Whether you’re a researcher, hobbyist, or just curious about neural sequence modeling, you’re in the right place.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Usage](#usage)
9. [Results](#results)
10. [Customization](#customization)
11. [Future Work](#future-work)
12. [Contributing](#contributing)
13. [License](#license)
14. [Contact](#contact)

---

## Project Overview

This project demonstrates how to build and train a character-level RNN for text generation using PyTorch. The goal is to learn patterns in a corpus (e.g., classic literature, song lyrics, code snippets) and generate new, coherent sequences in the same style.

Key motivations:

* Understand the mechanics of sequence modeling with RNNs.
* Experiment with hyperparameters like sequence length, hidden size, and learning rate.
* Explore how text generation quality evolves during training.

---

## Features

* **Character-level modeling:** Works at the granularity of individual characters.
* **Configurable architecture:** Easily adjust number of layers, hidden units, and dropout.
* **Checkpointing:** Save and load model weights during training.
* **Sample generation:** Generate text with variable temperature for creative control.
* **GPU support:** Automatically uses CUDA if available.

---

## Requirements

* Python 3.8+
* PyTorch 1.12+
* NumPy
* tqdm

> **Pro tip:** Virtual environments are your best friend—create one to avoid dependency headaches.

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/rnn-text-generation.git
   cd rnn-text-generation
   ```

2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset

1. Place your text file in the `data/` directory. The notebook expects a file named `input.txt`.
2. The data loader will automatically read and encode characters to integers.

> **Heads-up:** Make sure your text is clean (remove weird characters) for best results.

---

## Model Architecture

The core model is a simple `nn.RNN` (or `nn.LSTM`/`nn.GRU` if you switch) wrapped in a PyTorch `Module`. Key components:

* **Embedding layer:** Maps each character index to a dense vector.
* **Recurrent layer:** Processes sequential input; supports multiple layers and dropout.
* **Fully-connected layer:** Projects hidden state back to character logits.

Hyperparameters you can tweak:

* `hidden_size`: Dimensionality of RNN hidden state.
* `num_layers`: Number of stacked RNN layers.
* `dropout`: Dropout probability between RNN layers.
* `seq_length`: Number of characters per training sequence.
* `learning_rate`: Step size for optimizer.

---

## Training

1. Open and run `RNN.ipynb` in Google Colab or locally via Jupyter.
2. Configure hyperparameters in the first notebook cell.
3. Execute cells sequentially:

   * Data loading and preprocessing
   * Model definition
   * Training loop with loss logging and checkpointing
   * Text sampling at different temperatures

Training tips:

* Start small (`hidden_size=128`, `num_layers=1`) to make sure everything works.
* Gradually increase complexity to improve generation quality.
* Monitor the loss curve; if it plateaus too early, adjust the learning rate.

---

## Usage

After training, sample text using the saved checkpoint:

```bash
python sample.py --checkpoint checkpoints/model_epoch10.pt --temperature 0.8 --length 500
```

Parameters:

* `--checkpoint`: Path to saved model weights.
* `--temperature`: Sampling randomness (lower = conservative, higher = creative).
* `--length`: Number of characters to generate.

---

## Results

Here’s an example of text generated after 10 epochs at temperature 0.5:

> "She had not thought to find herself betrayed so soon — the moonlight
> danced upon the broken rails of the old railway track, echoing his promise
> in every hollow."

Not too shabby for a minimal RNN!

---

## Customization

* **Switch to LSTM/GRU:** Change `nn.RNN` to `nn.LSTM` or `nn.GRU` in the model class.
* **Word-level modeling:** Preprocess data at the word level and adjust vocabulary.
* **Attention mechanisms:** Add attention layers for smarter context handling.

---

## Future Work

* Implement bidirectional RNNs.
* Experiment with Transformer architectures for comparison.
* Build a simple web demo for interactive generation.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

Please follow the existing coding style and add tests where applicable.

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Contact

Developed by Nil (Studying Computer Engineering)

* GitHub: [NILatGit](https://github.com/NILatGit)
* Email: [nskarmakar.cse.ug@jadavpuruniversity.in](mailto:nskarmakar.cse.ug@jadavpuruniversity.in)

Happy experimenting!

