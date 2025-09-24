
# One Qubit Recurrent Network: Modeling Hidden States with Quantum Rotations

[](https://opensource.org/licenses/MIT)

This project introduces a **One Qubit Recurrent Neural Network (QRNN)**, a novel architecture that models the hidden states of an RNN using quantum rotations. It's a quantum-inspired approach designed for superior parameter efficiency in time-series analysis.

-----

### Research Poster

![poster](https://github.com/user-attachments/assets/dd29eb99-ef86-49a8-9347-027878432159)


-----

## Key Features

  - **Innovative Architecture**: Replaces classical weight matrices with a simple, trainable quantum circuit.
  - **Extreme Parameter Efficiency**: Achieves high performance with significantly fewer parameters than classical RNNs or LSTMs.
  - **Rich Expressiveness**: The QRNN's parameter space shows higher sensitivity and a more complex output landscape, indicating a greater expressive potential.

-----

## Architecture & Results

The QRNN encodes input and hidden state information into the rotation angles of quantum gates (**RY, RX, RZ**) on a single qubit. The final measurement of this qubit determines the network's output. This quantum mechanism replaces the traditional matrix multiplication in classical RNNs.

In performance tests on **stock price** and **sinusoidal data**, our QRNN (with only 4 parameters) consistently outperformed classical RNN and LSTM models.

  - On stock data, it achieved a **88.24% lower validation loss** than LSTM.
  - On sinusoidal data, it achieved a **99.99% lower validation loss** than LSTM.

This demonstrates the model's ability to learn complex temporal patterns with minimal resources.

-----

## Getting Started

### Prerequisites

  - `python >= 3.8`
  - `torch`
  - `qiskit` & `qiskit_machine_learning`
  - `yfinance`
  - `numpy`
  - `matplotlib`

### Installation & Usage

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/your-username/one-qubit-rnn.git
    cd one-qubit-rnn
    ```

2.  **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3.  **Run the experiments:**

      - Execute `main.py` to train and evaluate the models on the provided datasets.
      - Run `analysis.py` to generate the parameter space and distribution comparison plots.

-----

## Citation

If you find this work useful, please cite the authors:

**Seung Jik Kim, Jun Woo You, and Won Woe Ro**
*School of Electrical and Electronic Engineering, Yonsei University, Seoul, South Korea*
