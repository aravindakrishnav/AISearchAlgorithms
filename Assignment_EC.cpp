
//Neural Network
#include <iostream>
#include <cmath>

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double relu(double x) {
    return fmax(0.0, x);
}

double elu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * (exp(x) - 1);
}

double prelu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double leaky_relu(double x, double alpha = 0.01) {
    return x > 0 ? x : alpha * x;
}

double selu(double x, double lambda = 0.01) {
    return lambda * elu(x);
}

double softsign(double x) {
    return x / (1 + fabs(x));
}

double softplus(double x) {
    return log(1 + exp(x));
}

double hard_sigmoid(double x) {
    if (x >= 2.5) {
        return 1.0;
    } else if (x > -2.5 && x < 2.5) {
        return 0.2 * x + 0.5;
    } else {
        return 0.0;
    }
}

double swish(double x) {
    return x * sigmoid(x);
}

double mish(double x) {
    return x * tanh(log(1 + exp(x)));
}

int main() {
    // Example usage
    double input = 2.0;

    std::cout << "Sigmoid: " << sigmoid(input) << std::endl;
    std::cout << "Tanh: " << tanh(input) << std::endl;
    std::cout << "ReLU: " << relu(input) << std::endl;
    std::cout << "ELU: " << elu(input) << std::endl;
    std::cout << "PReLU: " << prelu(input) << std::endl;
    std::cout << "Leaky ReLU: " << leaky_relu(input) << std::endl;
    std::cout << "SELU: " << selu(input) << std::endl;
    std::cout << "Softsign: " << softsign(input) << std::endl;
    std::cout << "Softplus: " << softplus(input) << std::endl;
    std::cout << "Hard Sigmoid: " << hard_sigmoid(input) << std::endl;
    std::cout << "Swish: " << swish(input) << std::endl;
    std::cout << "Mish: " << mish(input) << std::endl;

    return 0;
}

//Custom Activation Function

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>

double custom_activation(double x) {
    
    return 1.0 / (1.0 + exp(-x));
}

double custom_activation_derivative(double x) {
    
    double fx = custom_activation(x);
    return fx * (1 - fx);
}

void initialize_parameters(int input_size, const std::vector<int>& hidden_layer_sizes, int output_size, std::vector<std::vector<double>>& weights) {
    std::srand(42);
    int total_layers = hidden_layer_sizes.size() + 2;
    weights.resize(total_layers);

    for (int i = 1; i < total_layers; ++i) {
        weights[i - 1].resize(hidden_layer_sizes[i - 1], i == total_layers - 1 ? output_size : hidden_layer_sizes[i]);
        
        for (int j = 0; j < weights[i - 1].size(); ++j) {
            for (int k = 0; k < weights[i - 1][j].size(); ++k) {
                weights[i - 1][j][k] = static_cast<double>(std::rand()) / RAND_MAX;
            }
        }
    }
}

std::vector<std::vector<double>> forward_propagation(const std::vector<double>& X, const std::vector<std::vector<double>>& weights) {
    int total_layers = weights.size() + 1;
    std::vector<std::vector<double>> layer_outputs(total_layers);
    layer_outputs[0] = X;

    for (int i = 1; i < total_layers; ++i) {
        std::vector<double> layer_input(weights[i - 1][0].size(), 0.0);

        for (int j = 0; j < weights[i - 1].size(); ++j) {
            for (int k = 0; k < weights[i - 1][j].size(); ++k) {
                layer_input[k] += layer_outputs[i - 1][j] * weights[i - 1][j][k];
            }
        }

        for (int j = 0; j < layer_input.size(); ++j) {
            layer_outputs[i].push_back(custom_activation(layer_input[j]));
        }
    }

    return layer_outputs;
}

std::vector<std::vector<double>> backward_propagation(const std::vector<double>& X, const std::vector<double>& y, const std::vector<std::vector<double>>& layer_outputs, const std::vector<std::vector<double>>& weights) {
    int total_layers = weights.size() + 1;
    std::vector<std::vector<double>> errors(total_layers - 1);
    std::vector<std::vector<double>> deltas(total_layers - 1);

    errors[total_layers - 2] = y;
    deltas[total_layers - 2] = std::vector<double>(y.size());

    for (int i = 0; i < y.size(); ++i) {
        deltas[total_layers - 2][i] = errors[total_layers - 2][i] * custom_activation_derivative(layer_outputs[total_layers - 1][i]);
    }

    for (int i = total_layers - 3; i >= 0; --i) {
        errors[i] = std::vector<double>(weights[i][0].size(), 0.0);
        for (int j = 0; j < weights[i + 1].size(); ++j) {
            for (int k = 0; k < weights[i + 1][j].size(); ++k) {
                errors[i][k] += deltas[i + 1][j] * weights[i + 1][j][k];
            }
        }

        deltas[i] = std::vector<double>(errors[i].size());
        for (int j = 0; j < errors[i].size(); ++j) {
            deltas[i][j] = errors[i][j] * custom_activation_derivative(layer_outputs[i + 1][j]);
        }
    }

    return deltas;
}

void update_weights(const std::vector<std::vector<double>>& layer_outputs, const std::vector<std::vector<double>>& deltas, std::vector<std::vector<double>>& weights, double learning_rate) {
    for (int i = 0; i < weights.size(); ++i) {
        for (int j = 0; j < weights[i].size(); ++j) {
            for (int k = 0; k < weights[i][j].size(); ++k) {
                weights[i][j][k] += layer_outputs[i][j] * deltas[i][k] * learning_rate;
            }
        }
    }
}

void train_neural_network(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int input_size, const std::vector<int>& hidden_layer_sizes, int output_size, int epochs, double learning_rate) {
    std::vector<std::vector<double>> weights;
    initialize_parameters(input_size, hidden_layer_sizes, output_size, weights);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (int sample = 0; sample < X.size(); ++sample) {
           
            std::vector<std::vector<double>> layer_outputs = forward_propagation(X[sample], weights);

            
            std::vector<std::vector<double>> deltas = backward_propagation(X[sample], y[sample], layer_outputs, weights);

            
            update_weights(layer_outputs, deltas, weights, learning_rate);
        }

        
        if (epoch % 100 == 0) {
            double mse = 0.0;
            for (int sample = 0; sample < X.size(); ++sample) {
                std::vector<std::vector<double>> predictions = forward_propagation(X[sample], weights);
                for (int i = 0; i < predictions.back().size(); ++i) {
                    mse += std::pow(y[sample][i] - predictions.back()[i], 2);
                }
            }
            mse /= X.size();

            std::cout << "Epoch " << epoch << ", Mean Squared Error: " << mse << std::endl;
        }
    }
}

int main() {
    // Example usage
    int input_size = 2;
    std::vector<int> hidden_layer_sizes = {4, 3}; 
    int output_size = 1;
    int epochs = 1000;
    double learning_rate = 0.1;

    std::vector<std::vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> y = {{0}, {1}, {1}, {0}};

    train_neural_network(X, y, input_size, hidden_layer_sizes, output_size, epochs, learning_rate);

    return 0;
}


//GWO

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Forward propagation through the neural network
std::vector<double> forwardPropagation(const std::vector<double>& X, const std::vector<std::vector<double>>& weights) {
    int inputSize = X.size();
    int hiddenSize = weights[0].size() - 1;
    int outputSize = weights[1][0].size() - 1;

    // Add bias to input layer
    std::vector<double> X_bias(X.begin(), X.end());
    X_bias.push_back(1.0);

    // Hidden layer
    std::vector<double> hiddenInput(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j <= inputSize; ++j) {
            hiddenInput[i] += X_bias[j] * weights[0][j][i];
        }
    }

    std::vector<double> hiddenOutput(hiddenSize);
    std::transform(hiddenInput.begin(), hiddenInput.end(), hiddenOutput.begin(), sigmoid);

    // Output layer
    std::vector<double> outputInput(outputSize, 0.0);
    outputInput[0] = 1.0; // Bias for the output layer
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j <= hiddenSize; ++j) {
            outputInput[i] += hiddenOutput[j] * weights[1][j][i];
        }
    }

    std::vector<double> output(outputSize);
    std::transform(outputInput.begin(), outputInput.end(), output.begin(), sigmoid);

    return output;
}

// Decode wolf position to neural network weights
std::vector<std::vector<std::vector<double>>> decodeWolfPosition(const std::vector<double>& X, int inputSize, int hiddenSize, int outputSize) {
    int inputHiddenSize = (inputSize + 1) * hiddenSize;
    int hiddenOutputSize = (hiddenSize + 1) * outputSize;

    std::vector<std::vector<std::vector<double>>> weights(2);
    weights[0].resize(inputSize + 1, std::vector<double>(hiddenSize));
    weights[1].resize(hiddenSize + 1, std::vector<double>(outputSize));

    for (int i = 0; i < inputHiddenSize; ++i) {
        weights[0][i % (inputSize + 1)][i / (inputSize + 1)] = X[i];
    }

    for (int i = inputHiddenSize; i < inputHiddenSize + hiddenOutputSize; ++i) {
        weights[1][(i - inputHiddenSize) % (hiddenSize + 1)][(i - inputHiddenSize) / (hiddenSize + 1)] = X[i];
    }

    return weights;
}

// Compute mean squared error fitness
double fitness(const std::vector<double>& X, const std::vector<std::vector<double>>& XTrain, const std::vector<std::vector<double>>& yTrain, int inputSize, int hiddenSize, int outputSize) {
    auto weights = decodeWolfPosition(X, inputSize, hiddenSize, outputSize);
    auto predictedOutput = forwardPropagation(XTrain[0], weights);
    double mse = 0.0;

    for (int i = 0; i < outputSize; ++i) {
        mse += std::pow(yTrain[0][i] - predictedOutput[i], 2);
    }

    mse /= outputSize;
    return mse;
}

// Grey Wolf Optimizer
std::vector<double> gwoOptimizer(const std::vector<std::vector<double>>& XTrain, const std::vector<std::vector<double>>& yTrain, int inputSize, int hiddenSize, int outputSize, int numWolves, int epochs) {
    int numDimensions = (inputSize + 1) * hiddenSize + (hiddenSize + 1) * outputSize;
    std::vector<std::vector<double>> wolvesPosition(numWolves, std::vector<double>(numDimensions));

    for (int i = 0; i < numWolves; ++i) {
        for (int j = 0; j < numDimensions; ++j) {
            wolvesPosition[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    auto alpha = wolvesPosition[0];
    auto beta = wolvesPosition[1];
    auto delta = wolvesPosition[2];

    double fitnessBeta = std::numeric_limits<double>::infinity();
    double fitnessDelta = std::numeric_limits<double>::infinity();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double a = 2.0 - 2.0 * epoch / epochs; // Linearly decrease from 2 to 0

        for (int i = 0; i < numWolves; ++i) {
            std::vector<double> A(numDimensions);
            std::vector<double> C(numDimensions);
            std::vector<double> DAlpha(numDimensions);
            std::vector<double> DBeta(numDimensions);
            std::vector<double> DDelta(numDimensions);

            // Generate random vectors
            for (int j = 0; j < numDimensions; ++j) {
                A[j] = 2.0 * a * static_cast<double>(std::rand()) / RAND_MAX - a;
                C[j] = 2.0 * static_cast<double>(std::rand()) / RAND_MAX;
                DAlpha[j] = std::abs(C[j] * alpha[j] - wolvesPosition[i][j]);
                DBeta[j] = std::abs(C[j] * beta[j] - wolvesPosition[i][j]);
                DDelta[j] = std::abs(C[j] * delta[j] - wolvesPosition[i][j]);
            }

            std::vector<double> X1(numDimensions);
            std::vector<double> X2(numDimensions);
            std::vector<double> X3(numDimensions);

            // Update positions
            for (int j = 0; j < numDimensions; ++j) {
                X1[j] = alpha[j] - A[j] * DAlpha[j];
                X2[j] = beta[j] - A[j] * DBeta[j];
                X3[j] = delta[j] - A[j

#include <iostream>
#include <vector>
#include <cmath>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Initialize neural network weights
std::vector<std::vector<std::vector<double>>> initializeWeights(int inputSize, int hiddenSize, int outputSize) {
    std::vector<std::vector<std::vector<double>>> weights(2);
    weights[0].resize(inputSize + 1, std::vector<double>(hiddenSize));
    weights[1].resize(hiddenSize + 1, std::vector<double>(outputSize));

    for (int i = 0; i < inputSize + 1; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            weights[0][i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i < hiddenSize + 1; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            weights[1][i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    return weights;
}

// Forward propagation through the neural network
std::vector<double> forwardPropagation(const std::vector<double>& X, const std::vector<std::vector<std::vector<double>>>& weights) {
    int inputSize = X.size();
    int hiddenSize = weights[0][0].size();
    int outputSize = weights[1][0].size();

    // Add bias to input layer
    std::vector<double> XBias(X.begin(), X.end());
    XBias.push_back(1.0);

    // Hidden layer
    std::vector<double> hiddenInput(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j <= inputSize; ++j) {
            hiddenInput[i] += XBias[j] * weights[0][j][i];
        }
    }

    std::vector<double> hiddenOutput(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenOutput[i] = sigmoid(hiddenInput[i]);
    }

    // Output layer
    std::vector<double> outputInput(outputSize, 0.0);
    outputInput[0] = 1.0; // Bias for the output layer
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j <= hiddenSize; ++j) {
            outputInput[i] += hiddenOutput[j] * weights[1][j][i];
        }
    }

    std::vector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = sigmoid(outputInput[i]);
    }

    return output;
}

// Train neural network using GWO-optimized weights
std::vector<std::vector<std::vector<double>>> trainNeuralNetwork(const std::vector<std::vector<double>>& XTrain, const std::vector<std::vector<double>>& yTrain, int inputSize, int hiddenSize, int outputSize, int epochs, double learningRate) {
    // Initialize weights
    auto weights = initializeWeights(inputSize, hiddenSize, outputSize);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward propagation
        auto hiddenInput = forwardPropagation(XTrain[0], weights[0]);
        auto hiddenOutput = forwardPropagation(hiddenInput, weights[1]);

        // Backpropagation
        auto outputError = yTrain[0][0] - hiddenOutput[0];
        auto outputDelta = outputError * hiddenOutput[0] * (1 - hiddenOutput[0]);

        auto hiddenError = outputDelta * weights[1][1][0]; // Assuming one neuron in the hidden layer
        auto hiddenDelta = hiddenError * hiddenInput[0] * (1 - hiddenInput[0]);

        // Update weights
        for (int i = 0; i <= inputSize; ++i) {
            weights[0][i][0] += learningRate * XTrain[0][i] * hiddenDelta;
        }

        for (int i = 0; i <= hiddenSize; ++i) {
            weights[1][i][0] += learningRate * hiddenOutput[i] * outputDelta;
        }

        if (epoch % 1000 == 0) {
            double mse = std::pow(yTrain[0][0] - hiddenOutput[0], 2);
            std::cout << "Epoch " << epoch << ", MSE: " << mse << std::endl;
        }
    }

    return weights;
}

int main() {
    // Example usage
    int inputSize = 2;
    int hiddenSize = 4;
    int outputSize = 1;

    std::vector<std::vector<double>> XTrain = {{0, 0}};
    std::vector<std::vector<double>> yTrain = {{0}};

    int epochs = 10000;
    double learningRate = 0.1;

    // Train the neural network using GWO-optimized weights
    auto gwoOptimizedWeights = trainNeuralNetwork(XTrain, yTrain, inputSize, hiddenSize, outputSize, epochs, learningRate);

    // Use the GWO-optimized weights for making predictions
    auto predictedOutput = forwardPropagation(XTrain[0], gwoOptimizedWeights);

    // Print the results
    std::cout << "Predicted Output using GWO-optimized weights:" << std::endl;
    for (double output : predictedOutput) {
        std::cout << output << " ";
    }

    return 0;
}

//Genetic Algorithm

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Neural network weights structure
struct Chromosome {
    std::vector<std::vector<double>> inputHidden;
    std::vector<std::vector<double>> hiddenOutput;
};

// Initialize neural network weights
Chromosome initializeWeights(int inputSize, int hiddenSize, int outputSize) {
    Chromosome chromosome;
    chromosome.inputHidden.resize(inputSize + 1, std::vector<double>(hiddenSize));
    chromosome.hiddenOutput.resize(hiddenSize + 1, std::vector<double>(outputSize));

    for (int i = 0; i <= inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            chromosome.inputHidden[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i <= hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            chromosome.hiddenOutput[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    return chromosome;
}

// Forward propagation through the neural network
std::vector<double> forwardPropagation(const std::vector<double>& X, const Chromosome& chromosome) {
    int inputSize = X.size();
    int hiddenSize = chromosome.inputHidden[0].size();
    int outputSize = chromosome.hiddenOutput[0].size();

    // Add bias to input layer
    std::vector<double> XBias(X.begin(), X.end());
    XBias.push_back(1.0);

    // Hidden layer
    std::vector<double> hiddenInput(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j <= inputSize; ++j) {
            hiddenInput[i] += XBias[j] * chromosome.inputHidden[j][i];
        }
    }

    std::vector<double> hiddenOutput(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenOutput[i] = sigmoid(hiddenInput[i]);
    }

    // Output layer
    std::vector<double> outputInput(outputSize, 0.0);
    outputInput[0] = 1.0; // Bias for the output layer
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j <= hiddenSize; ++j) {
            outputInput[i] += hiddenOutput[j] * chromosome.hiddenOutput[j][i];
        }
    }

    std::vector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = sigmoid(outputInput[i]);
    }

    return output;
}

// Genetic Algorithm Functions
std::vector<Chromosome> initializePopulation(int populationSize, int inputSize, int hiddenSize, int outputSize) {
    std::vector<Chromosome> population;
    for (int i = 0; i < populationSize; ++i) {
        population.push_back(initializeWeights(inputSize, hiddenSize, outputSize));
    }
    return population;
}

double calculateFitness(const Chromosome& chromosome, const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) {
    auto predictedOutput = forwardPropagation(X[0], chromosome);
    double mse = 0.0;

    for (int i = 0; i < y[0].size(); ++i) {
        mse += std::pow(y[0][i] - predictedOutput[i], 2);
    }

    mse /= y[0].size();
    return 1.0 / (1.0 + mse); // Maximizing fitness, so using the inverse of MSE
}

Chromosome crossover(const Chromosome& parent1, const Chromosome& parent2) {
    Chromosome child;
    for (int i = 0; i < parent1.inputHidden.size(); ++i) {
        int crossoverPoint = rand() % parent1.inputHidden[i].size();
        child.inputHidden[i] = std::vector<double>(parent1.inputHidden[i].begin(), parent1.inputHidden[i].begin() + crossoverPoint);
        child.inputHidden[i].insert(child.inputHidden[i].end(), parent2.inputHidden[i].begin() + crossoverPoint, parent2.inputHidden[i].end());
    }

    for (int i = 0; i < parent1.hiddenOutput.size(); ++i) {
        int crossoverPoint = rand() % parent1.hiddenOutput[i].size();
        child.hiddenOutput[i] = std::vector<double>(parent1.hiddenOutput[i].begin(), parent1.hiddenOutput[i].begin() + crossoverPoint);
        child.hiddenOutput[i].insert(child.hiddenOutput[i].end(), parent2.hiddenOutput[i].begin() + crossoverPoint, parent2.hiddenOutput[i].end());
    }

    return child;
}

Chromosome mutate(const Chromosome& chromosome, double mutationRate) {
    Chromosome mutatedChromosome;
    for (int i = 0; i < chromosome.inputHidden.size(); ++i) {
        mutatedChromosome.inputHidden[i] = chromosome.inputHidden[i];
        for (int j = 0; j < mutatedChromosome.inputHidden[i].size(); ++j) {
            if (rand() / static_cast<double>(RAND_MAX) < mutationRate) {
                mutatedChromosome.inputHidden[i][j] += (rand() / static_cast<double>(RAND_MAX)) * 0.2 - 0.1;
            }
        }
    }

    for (int i = 0; i < chromosome.hiddenOutput.size(); ++i) {
        mutatedChromosome.hiddenOutput[i] = chromosome.hiddenOutput[i];
        for (int j = 0; j < mutatedChromosome.hiddenOutput[i].size(); ++j) {
            if (rand() / static_cast<double>(RAND_MAX) < mutationRate) {
                mutatedChromosome.hiddenOutput[i][j] += (rand() / static_cast<double>(RAND_MAX)) * 0.2 - 0.1;
            }
        }
    }

    return mutatedChromosome;
}

Chromosome geneticAlgorithm(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int inputSize, int hiddenSize, int outputSize, int populationSize, int generations, double crossoverRate, double mutationRate) {
    // Initialize population
    std::vector<Chromosome> population = initializePopulation(populationSize, inputSize, hiddenSize, outputSize);

    for (int generation = 0; generation < generations; ++generation) {
        std::vector<double> fitnessScores;
        for (const auto& chromosome : population) {
            fitnessScores.push_back(calculateFitness(chromosome, X, y));
        }

        // Select parents based on fitness
        std::vector<Chromosome> parents;
        for (int i = 0; i < populationSize; ++i) {
            int index = rand() % populationSize;
            parents.push_back(population[index]);
        }

        // Create the next generation through crossover and mutation
        std::vector<Chromosome> nextGeneration;
        for (int i = 0; i < populationSize; ++i) {
            Chromosome parent1 = parents[rand() % populationSize];
            Chromosome parent2 = parents[rand() % populationSize];
            Chromosome child = (rand() / static_cast<double>(RAND_MAX) < crossoverRate) ? crossover(parent1, parent2) : parent1;
            child = mutate(child, mutationRate);
            nextGeneration.push_back(child);
        }

        population = nextGeneration;

        if (generation % 10 == 0) {
            double bestFitness = calculateFitness(population[0], X, y);
            std::cout << "Generation " << generation << ", Best Fitness: " << bestFitness << std::endl;
        }
    }

    return population[0];
}

int main() {
    // Example usage
    int inputSize = 2;
    int hiddenSize = 5;
    int outputSize = 1;

    std::vector<std::vector<double>> XTrain = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 1}, {0, 1}, {0, 1}};
    std::vector<std::vector<double>> yTrain = {{0}, {1}, {1}, {0}, {0}, {1}, {0}};

    int populationSize = 20;
    int generations = 100;
    double crossoverRate = 0.8;
    double mutationRate = 0.1;

    Chromosome bestIndividual = geneticAlgorithm(XTrain, yTrain, inputSize, hiddenSize, outputSize, populationSize, generations, crossoverRate, mutationRate);

    std::vector<double> bestInputHiddenWeights = forwardPropagation(XTrain[0], bestIndividual);
    std::vector<double> bestHiddenOutputWeights = forwardPropagation(bestInputHiddenWeights, bestIndividual);

    std::cout << "Best Predicted Output:" << std::endl;
    for (double output : bestHiddenOutputWeights) {
        std::cout << output << " ";
    }

    return 0;
}

//Particle Swarm Optimization

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

// Neural network weights structure
struct Particle {
    std::vector<std::vector<double>> inputHidden;
    std::vector<std::vector<double>> hiddenOutput;
    std::vector<std::vector<double>> velocityInputHidden;
    std::vector<std::vector<double>> velocityHiddenOutput;
    std::vector<std::vector<double>> personalBestInputHidden;
    std::vector<std::vector<double>> personalBestHiddenOutput;
    double personalBestFitness;
};

// Initialize neural network weights and velocities
Particle initializeParticle(int inputSize, int hiddenSize, int outputSize) {
    Particle particle;
    particle.inputHidden.resize(inputSize, std::vector<double>(hiddenSize));
    particle.hiddenOutput.resize(hiddenSize, std::vector<double>(outputSize));
    particle.velocityInputHidden.resize(inputSize, std::vector<double>(hiddenSize, 0.0));
    particle.velocityHiddenOutput.resize(hiddenSize, std::vector<double>(outputSize, 0.0));
    particle.personalBestInputHidden = particle.inputHidden;
    particle.personalBestHiddenOutput = particle.hiddenOutput;
    particle.personalBestFitness = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < hiddenSize; ++j) {
            particle.inputHidden[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            particle.hiddenOutput[i][j] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    return particle;
}

// Forward propagation through the neural network
std::vector<double> forwardPropagation(const std::vector<double>& X, const Particle& particle) {
    int inputSize = X.size();
    int hiddenSize = particle.inputHidden[0].size();
    int outputSize = particle.hiddenOutput[0].size();

    // Hidden layer
    std::vector<double> hiddenInput(hiddenSize, 0.0);
    for (int i = 0; i < hiddenSize; ++i) {
        for (int j = 0; j <= inputSize; ++j) {
            hiddenInput[i] += (j < inputSize) ? X[j] * particle.inputHidden[j][i] : 1.0; // Add bias term
        }
    }

    std::vector<double> hiddenOutput(hiddenSize);
    for (int i = 0; i < hiddenSize; ++i) {
        hiddenOutput[i] = sigmoid(hiddenInput[i]);
    }

    // Output layer
    std::vector<double> outputInput(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j <= hiddenSize; ++j) {
            outputInput[i] += (j < hiddenSize) ? hiddenOutput[j] * particle.hiddenOutput[j][i] : 1.0; // Add bias term
        }
    }

    std::vector<double> output(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        output[i] = sigmoid(outputInput[i]);
    }

    return output;
}

// Calculate fitness based on mean squared error
double calculateFitness(const Particle& particle, const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) {
    auto predictedOutput = forwardPropagation(X[0], particle);
    double mse = 0.0;

    for (int i = 0; i < y[0].size(); ++i) {
        mse += std::pow(y[0][i] - predictedOutput[i], 2);
    }

    mse /= y[0].size();
    return 1.0 / (1.0 + mse); // Maximizing fitness, so using the inverse of MSE
}

// Update particle velocity based on PSO formula
void updateParticleVelocity(Particle& particle, const Particle& globalBest, double inertiaWeight, double cognitiveWeight, double socialWeight) {
    double rand1 = static_cast<double>(std::rand()) / RAND_MAX;
    double rand2 = static_cast<double>(std::rand()) / RAND_MAX;

    for (int i = 0; i < particle.velocityInputHidden.size(); ++i) {
        for (int j = 0; j < particle.velocityInputHidden[i].size(); ++j) {
            particle.velocityInputHidden[i][j] = inertiaWeight * particle.velocityInputHidden[i][j] +
                                                  cognitiveWeight * rand1 * (particle.personalBestInputHidden[i][j] - particle.inputHidden[i][j]) +
                                                  socialWeight * rand2 * (globalBest.inputHidden[i][j] - particle.inputHidden[i][j]);
        }
    }

    for (int i = 0; i < particle.velocityHiddenOutput.size(); ++i) {
        for (int j = 0; j < particle.velocityHiddenOutput[i].size(); ++j) {
            particle.velocityHiddenOutput[i][j] = inertiaWeight * particle.velocityHiddenOutput[i][j] +
                                                   cognitiveWeight * rand1 * (particle.personalBestHiddenOutput[i][j] - particle.hiddenOutput[i][j]) +
                                                   socialWeight * rand2 * (globalBest.hiddenOutput[i][j] - particle.hiddenOutput[i][j]);
        }
    }
}

// Update particle position based on PSO formula
void updateParticlePosition(Particle& particle, const std::vector<double>& bounds) {
    for (int i = 0; i < particle.inputHidden.size(); ++i) {
        for (int j = 0; j < particle.inputHidden[i].size(); ++j) {
            particle.inputHidden[i][j] += particle.velocityInputHidden[i][j];
            particle.inputHidden[i][j] = std::max(std::min(particle.inputHidden[i][j], bounds[0]), bounds[1]);
        }
    }

    for (int i = 0; i < particle.hiddenOutput.size(); ++i) {
        for (int j = 0; j < particle.hiddenOutput[i].size(); ++j) {
            particle.hiddenOutput[i][j] += particle.velocityHiddenOutput[i][j];
            particle.hiddenOutput[i][j] = std::max(std::min(particle.hiddenOutput[i][j], bounds[0]), bounds[1]);
        }
    }
}

// Particle Swarm Optimization algorithm
Particle particleSwarmOptimization(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y, int inputSize, int hiddenSize, int outputSize, int numParticles, int numIterations, double inertiaWeight, double cognitiveWeight, double socialWeight) {
    std::vector<double> bounds = {0, 1}; // Bounds for weight values
    std::vector<Particle> particles;

    // Initialize particles
    for (int i = 0; i < numParticles; ++i) {
        particles.push_back(initializeParticle(inputSize, hiddenSize, outputSize));
    }

    Particle globalBest;
    globalBest.personalBestFitness = -std::numeric_limits<double>::infinity();

    for (int iteration = 0; iteration < numIterations; ++iteration) {
        for (auto& particle : particles) {
            double fitness = calculateFitness(particle, X, y);

            // Update personal best
            if (fitness > particle.personalBestFitness) {
                particle.personalBestInputHidden = particle.inputHidden;
                particle.personalBestHiddenOutput = particle.hiddenOutput;
                particle.personalBestFitness = fitness;
            }

            // Update global best
            if (fitness > globalBest.personalBestFitness) {
                globalBest.inputHidden = particle.inputHidden;
                globalBest.hiddenOutput = particle.hiddenOutput;
                globalBest.personalBestFitness = fitness;
            }
        }

        // Update particle velocities and positions
        for (auto& particle : particles) {
            updateParticleVelocity(particle, globalBest, inertiaWeight, cognitiveWeight, socialWeight);
            updateParticlePosition(particle, bounds);
        }

        if (iteration % 10 == 0) {
            std::cout << "Iteration " << iteration + 1 << ", Best Fitness: " << globalBest.personalBestFitness << std::endl;
        }
    }

    return globalBest;
}

int main() {
    // Example usage
    int inputSize = 2;
    int hiddenSize = 5;
    int outputSize = 1;

    std::vector<std::vector<double>> XTrain = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1, 1}, {0, 1}, {0, 1}};
    std::vector<std::vector<double>> yTrain = {{0}, {1}, {1}, {0}, {0}, {1}, {0}};

    int numParticles = 20;
    int numIterations = 100;
    double inertiaWeight = 0.5;
    double cognitiveWeight = 2.0;
    double socialWeight = 2.0;

    Particle bestParticle = particleSwarmOptimization(XTrain, yTrain, inputSize, hiddenSize, outputSize, numParticles, numIterations, inertiaWeight, cognitiveWeight, socialWeight);

    std::vector<double> bestInputHiddenWeights = forwardPropagation(XTrain[0], bestParticle);
    std::vector<double> bestHiddenOutputWeights = forwardPropagation(bestInputHiddenWeights, bestParticle);

    std::cout << "Best Predicted Output:" << std::endl;
    for (double output : bestHiddenOutputWeights) {
        std::cout << output << " ";
    }

    return 0;
}


