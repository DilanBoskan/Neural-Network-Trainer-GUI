import numpy as np
# System Modules
import sys
import os
import matplotlib as plt

# Change the current working directory to the directory
# this file sits in
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_path)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


class Feedforward_Neural_Network:
    def __init__(self, inputs: int, hidden: list, outputs: int):
        # Create Variables
        self.weights = []
        self.biases = []
        self.num_inputs = inputs
        self.num_hidden = np.array(hidden) if hidden is not None else []
        self.num_outputs = outputs
        self.num_layers = len(self.num_hidden) + 1

        self.randomize_WB()

    def randomize_WB(self, low=0., high=1.):
        """
        Randomize all weights and biases on a value between low and high
        """
        inp_hid_out = np.array([self.num_inputs, *self.num_hidden, self.num_outputs])  # nopep8
        self.weights = np.empty(shape=self.num_layers, dtype=np.object)
        self.biases = np.empty(shape=self.num_layers, dtype=np.object)
        """
        Every Iteration is corresponding to one layer:
            input_nums -> number of neurons on the left side of the layer
            output_nums -> number of neurons on the right side of the layer

        Example:
            layer1 = [[w00, w01, w02],
                      [w10, w11, w12]]

            layer1 has 2 inputs and 3 corresponding outputs
            layer[0] returns w00, w01, w02:
                    w00
                  +----> o0
                  | w01
            i0----+----> o1
                  | w02
                  +----> o2
        """
        for i, layer_nums in enumerate(zip(inp_hid_out[0:-1], inp_hid_out[1:])):

            input_nums = layer_nums[0]
            output_nums = layer_nums[1]

            self.weights[i] = np.random.uniform(
                size=(input_nums, output_nums), low=low, high=high)
            self.biases[i] = np.random.uniform(
                size=(output_nums), low=low, high=high)

    def load_WB(self, loc):
        """
        Load the weights and biases from the given location
        """
        array = np.load(loc, allow_pickle=True)

        self.weights = np.array(array[0])
        self.biases = np.array(array[1])

    def save_WB(self, loc):
        """
        Save the weights and biases at the given directory
        """
        # -Determine File Name-
        array = np.array([self.weights, self.biases])
        np.save(loc, array,
                allow_pickle=True)

    def get_Predictions(self, _inputs):
        activations = []
        layer_out = _inputs
        activations.append(layer_out)
        for cur_layer in range(self.num_layers):
            layer_in = np.dot(layer_out, self.weights[cur_layer])
            layer_in += self.biases[cur_layer]
            layer_out = sigmoid(layer_in)
            activations.append(layer_out)

        """
        activations is an array containg all activations of all neurons including the inputs
        activations[0] -> activations of the input neurons / input
        activations[1] -> activations of the next layer
        activations[-1] -> activations of the output layer
        """
        return activations

    def backpropogation(self, inputs: list, activations: list, labels: list, lr: float):
        """
        Perform the backpropogation and the return the new weights and biases
        """
        num_elements = np.size(labels)
        outputs = activations[-1]

        new_weights = self.weights.copy()
        new_biases = self.biases.copy()

        # Going from back to front
        # 1 -> output layer
        # until
        # j -> input layer
        cur_layer = 1

        gamma = 2 * (outputs - labels) / num_elements  # A

        for x in range(self.num_layers):
            # Current Layer Activasion
            cur_act = activations[-cur_layer]

            beta = gamma * sigmoid_derivative(cur_act)  # B
            new_biases[x] = np.sum(beta, axis=0, keepdims=True)
            delta = np.dot(activations[-cur_layer - 1].T, beta)  # D
            new_weights[x] = delta
            gamma = np.dot(beta, self.weights[-cur_layer].T)  # C
            cur_layer += 1

        # 'reverse' because we added everything backwards
        new_weights = np.flip(new_weights, axis=0)
        new_biases = np.flip(new_biases, axis=0)

        return new_weights, new_biases

    def mse(self, outputs, labels):
        """
        Calculate the Mean Square Error (MSE)
        """
        num_elements = np.size(labels)

        return np.sum((outputs - labels)**2) / num_elements

    def perform_training(self, training_inputs: np.ndarray, training_labels: np.ndarray, epochs: int = 1000, lr: float = 0.1, shuffle: bool = False, graph_data: dict = None):
        """
        Train the neural network

        Paramaters:
            epochs:
                How many times to train on the training data
            lr:
                Learning Rate
            shuffle:
                Whether to shuffle the training data/labels on every iteration
            graph:
                Whether to show a graph displaying the progress over the epochs
        """
        def unison_shuffled_copies(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]

        training_inputs = np.array(training_inputs)
        training_labels = np.array(training_labels)

        epoch_range = []
        errors = []
        annotation = None
        for child in graph_data['axis'].get_children():
            if isinstance(child, plt.text.Annotation):
                child.remove()
        print("Training...")
        for epoch in range(epochs):
            if shuffle:
                training_inputs, training_labels = unison_shuffled_copies(training_inputs, training_labels)  # nopep8
            # Get the activations for each neuron (including input)
            activations = self.get_Predictions(training_inputs)

            # Get the derivatives (slopes) of all weights and biases with repect to the MSE
            weight_slopes, bias_slopes = self.backpropogation(inputs=training_inputs,
                                                              activations=activations,
                                                              labels=training_labels,
                                                              lr=lr)

            # Update Weights
            self.weights = self.weights - lr*weight_slopes
            self.biases = self.biases - lr*bias_slopes
            
            epoch_range.append(epoch)
            errors.append(self.mse(activations[-1], training_labels))
            if not epoch % int(epochs / 10):
                graph_data['line'].set_xdata(epoch_range)
                graph_data['line'].set_ydata(errors)
                graph_data['figure'].canvas.draw()
                graph_data['figure'].canvas.flush_events()

                if annotation is not None:
                    annotation.remove()

                annotation = graph_data['axis'].annotate('%0.3f' % errors[-1], xy=(1, errors[-1]), xytext=(8, 0),
                                                         xycoords=('axes fraction', 'data'), textcoords='offset points')
        else:
            graph_data['line'].set_xdata(epoch_range)
            graph_data['line'].set_ydata(errors)
            graph_data['figure'].canvas.draw()
            graph_data['figure'].canvas.flush_events()

            annotation = graph_data['axis'].annotate('%0.3f' % errors[-1], xy=(1, errors[-1]), xytext=(8, 0),
                                                     xycoords=('axes fraction', 'data'), textcoords='offset points')

        print("MSE: {0}\n".format(self.mse(activations[-1], training_labels)))
        return errors[-1]

    def perform_test(self, training_inputs: np.ndarray):
        """
        Perform a test on the given training inputs
        """
        training_inputs = np.array(training_inputs)
        if (training_inputs.ndim == 1):
            training_inputs = np.array(training_inputs, ndmin=2)

        outputs = self.get_Predictions(training_inputs)[-1]

        return outputs

    def generate_modelStructure(self):
        """
        Print the Neural Network structure
        """
        model = str(self.num_inputs)
        for layer in self.num_hidden:
            model += " > {}".format(layer)
        model += " > {}".format(self.num_outputs)

        return model

    def generate_fileName(self):
        """
        Generate an appropiate file name based on the neural
        networks structure
        """
        fileName = ''
        for layer in [self.num_inputs, *self.num_hidden, self.num_outputs]:
            fileName += "_{0}".format(layer)
        fileName = 'Weights_Biases%s' % fileName
        return fileName


if __name__ == "__main__":
    fnn = Feedforward_Neural_Network(inputs=784,
                                     hidden=[50, 30], outputs=10)
    fnn.load_WB(loc='Weights_Biases_50_30.npy')

    training_data = np.load(r'B:\boska\Documents\Dilan\MNIST Data Set\Training_Data.npy',
                            allow_pickle=True)
    training_labels = np.load(r'B:\boska\Documents\Dilan\MNIST Data Set\Training_Labels.npy',
                              allow_pickle=True)

    fnn.perform_training(training_inputs=training_data,
                         training_labels=training_labels,
                         epochs=100,
                         lr=10000,)

    fnn.save_WB('')
