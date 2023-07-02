from abc import ABC
import random
import math
import time


def table_print(values, row_length=15):
    return_string = ""

    for val in values:
        val = str(val)
        if len(val) > row_length:
            return_string += val[:row_length]
        else:
            return_string += val + " "*(row_length-len(val))

    return return_string


def activation_function(value, activation_type):
    if activation_type == "sigmoid":
        return 1 / (1 + math.e ** (-value))

    elif activation_type == "tanh":
        return 2 / (1 + math.e ** (-2 * value)) - 1

    elif activation_type == "relu":
        return max(0, value)

    elif activation_type == "identity":
        return value

    elif activation_type == "binary":
        return 1 if value >= 1 else 0

    return None


class Neuron(ABC):
    def __init__(self):
        self.value = 0  # Aka Activation

    def __str__(self):
        return table_print(["Neuron", self.value])


class Layer:
    layer_count = 1

    def __init__(self, num_neurons, activation="identity", name=None):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.activation = activation
        self.weights = None
        self.biases = None

        # Set layer name
        if name is None:
            self.name = "Layer " + str(Layer.layer_count)
        else:
            self.name = name
        Layer.layer_count += 1

    def compile(self, prev_num_neurons, initial_weight=None, initial_bias=0):
        # Set weights
        if initial_weight is None:
            self.weights = [[random.random() for _ in range(prev_num_neurons)] for _ in range(len(self.neurons))]
        else:
            self.weights = [[initial_weight for _ in range(prev_num_neurons)] for _ in range(len(self.neurons))]

        # Set bias
        self.biases = [initial_bias for _ in range(len(self.neurons))]

    # Input values = neuron values of previous layer
    def calc_activation(self, input_values):
        for i, neuron in enumerate(self.neurons):
            weighted_values = [input_values[j] * self.weights[i][j] for j in range(len(input_values))]
            activation = activation_function(sum(weighted_values) + self.biases[i], self.activation)
            self.neurons[i].value = activation

    def __str__(self):
        if self.weights is not None and self.biases is not None:
            num_weights = len(self.weights[0]*len(self.weights)) + len(self.biases)
        else:
            num_weights = "Not compiled"
        return table_print([self.name, len(self.neurons), self.activation, num_weights])


class InputLayer:
    def __init__(self, num_neurons):
        self.neurons = [Neuron() for _ in range(num_neurons)]
        self.weights = None

    def compile(self, _):
        pass

    def calc_activation(self, _):
        pass

    def __str__(self):
        return table_print(["Input Layer", len(self.neurons)])


class MultilayerPerceptron:
    def __init__(self, num_input_neurons, name="Unnamed MLP"):
        self.layers = [InputLayer(num_input_neurons)]
        self.name = name
        self.compiled = False

    def add_layer(self, num_neurons, activation="identity", name=None):
        if not self.compiled:
            new_layer = Layer(num_neurons, activation=activation, name=name)
            self.layers.append(new_layer)
        else:
            print("Can't add layer to compiled model!")

    def compile(self):
        if not self.compiled:
            start_time = time.time()

            for i, layer in enumerate(self.layers[1:]):  # Skip Input Layer
                layer.compile(len(self.layers[i].neurons))
            self.compiled = True
            print(f"Compiled {self.name} in {round(time.time()-start_time, 4)} seconds")
        else:
            print("Model was already compiled!")

    def get_all_weights(self):
        all_weights = []
        for layer in self.layers[1:]:
            all_weights += [weight for weights in layer.weights for weight in weights]
        return all_weights

    def train(self, input_values, expected_output):
        model_out = self(input_values)
        cost = sum([(model_out[i]-expected_output[i])**2 for i in range(len(model_out))])  # SSE
        all_weights = self.get_all_weights()
        print(len(all_weights))

    def __call__(self, input_values):
        if self.compiled:
            for i, neuron in enumerate(self.layers[0].neurons):
                neuron.value = input_values[i]

            for i, layer in enumerate(self.layers[1:]):  # Skip Input Layer
                layer.calc_activation([neuron.value for neuron in self.layers[i].neurons])

            return [neuron.value for neuron in self.layers[-1].neurons]

        else:
            print("Model needs to be compiled first!")

    def __str__(self):
        cols = ["Type", "Neurons", "Activation", "Weights & Bias"]
        dash_len = len(cols) * 15
        return_str = "\n" + self.name + "\n" + "_"*dash_len + "\n" + table_print(cols) + "\n" + "-"*dash_len
        for layer in self.layers:
            return_str += "\n" + layer.__str__()

        return_str += "\n" + "-"*dash_len + "\n" + \
                      f"Total Neurons: {sum([len(layer.neurons) for layer in self.layers])}" + " "*5 + \
                      f"Total Weights: WIP" + "\n"

        return return_str


if __name__ == "__main__":

    mlp = MultilayerPerceptron(4)
    mlp.add_layer(8, activation="sigmoid")
    mlp.add_layer(16, activation="sigmoid")
    mlp.add_layer(3)
    mlp.compile()
    print(mlp)

    # print(mlp([random.random() for _ in range(4)]))
    print(mlp.train([random.random() for _ in range(4)], [0, 1, 0]))

