# ANN-Number-Recognition

A simple artificial neural network (ANN) implementation for recognizing handwritten digits using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

## Project Structure

```
ANN-Number-Recognition/
├── model.py             # Neural network model implementation
├── utils.py            # Helper functions
├── train.py            # Training script
├── test.py             # Testing and visualization
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training the network

```bash
python train.py
```

This will train the model on the MNIST dataset and save the parameters in a file defaultly named as `mnist.pth`.

### Testing / Inference

```bash
python test.py
```

This will evaluate the model on test samples and visualize predictions.


## Acknowledgements

* MNIST dataset from Yann LeCun's [website](http://yann.lecun.com/exdb/mnist/)

## License

This project is licensed under the MIT License.
