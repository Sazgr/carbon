# Carbon

Carbon is an open-source Neural Network (NN) trainer for chess engines, designed primarily to support Rice's neural networks , inspired by Grapheus, a previous NN trainer used to train Rice's neural networks.

<p align="center">
  <img src="logo.png" alt="Project Logo" width="30%">
</p>

## Features

- **Learning Rate Schedulers:** Utilize advanced learning rate scheduling techniques, including Step Decay and Cosine Annealing.
- **Optimizers:** Implement various optimization algorithms like SGD, Adam, Adamax to your preferences.
- **Configuration Made Easy:** Easily customize training parameters and model architecture to suit your needs.
- **Visualize Training:** Monitor training progress and loss using the included `lossplot.py` script.

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rafid-dev/carbon.git
   cd carbon
   ```

2. **Building:**
   ```bash
   make
   ./bin/CarbonTrainer
   ```

3. **Start Training:**
   Follow the provided instructions to start training your neural networks and improving your chess engine's evaluation capabilities. (TODO)