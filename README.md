# ProteinClassifier
## The Neural Network
This network predicts the proteins of an amino acid sequence based on an initial codon list and based on a built in dataset. The network is fully densely connected
and uses a sparse categorical crossentropy loss function since the dataset is categorical (the model predicts one of the possible protein classes). The neural network
uses a standard Adam optimizer with a learning rate of 0.001. The model's architecture contains:
- 1 Input Layer (with 3 input neurons)
- 12 Hidden Layers (with either 128, 256, or 1024 neurons and a standard ReLU activation function)
- 1 Output Layer (with 22 output neurons)

Feel free to hyperparameter tune the model or experiment with the dataset!

## The Dataset
The data consists of 

## Libraries
This neural network was created with the help of the Tensorflow library.
- Tensorflow's Website: https://www.tensorflow.org/
- Tensorflow Installation Instructions: https://www.tensorflow.org/install
