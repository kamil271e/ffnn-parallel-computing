#include "../lib/net.hpp"

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int num_classes, double lr){
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->num_classes = num_classes;
    this->lr = lr;
    this->accurate_pred = 0;
    init_weights();
}

void NeuralNetwork::init_weights(){
    hidden_weights = Tensor(hidden_size, input_size);
    hidden_weights.initUniform();

    output_weights = Tensor(num_classes, hidden_size);
    output_weights.initNorm();
}

double NeuralNetwork::fit(std::vector<Digit> digits){
    for(const Digit& digit: digits){
        forward_propagation(digit);
        backward_propagation(digit.label);
    }
    double accuracy = (double)accurate_pred  / (double)digits.size();
    return accuracy;
}

double NeuralNetwork::predict(std::vector<Digit> digits){
    accurate_pred = 0;
    for(const Digit& digit: digits){
        forward_propagation(digit);
    }
    double accuracy = (double)accurate_pred  / (double)digits.size();
    return accuracy;
}

void NeuralNetwork::forward_propagation(Digit digit){
    inputs = digit.data;
    inputs.flatten();

    hidden_outputs = hidden_weights * inputs;
    hidden_outputs.relu();

    outputs = output_weights * hidden_outputs;
    outputs.softmax();

    if (digit.label == outputs.argmax()) accurate_pred++;
}

void NeuralNetwork::backward_propagation(int target){
    labels = Tensor(num_classes, 1);
    labels.oneHotEncoding(target);

    output_err = labels - outputs;

    output_weights.transpose();

    hidden_err = output_weights * output_err;

    outputs.softmaxDerivative();

    output_gradients = outputs & output_err;

    hidden_outputs.transpose();

    output_gradients = output_gradients * hidden_outputs;

    output_gradients = output_gradients * lr;

    output_weights.transpose();

    output_weights = output_weights + output_gradients;

    ///////////////////////////////////////////////////

    hidden_outputs.reluDerivative();

    hidden_outputs.transpose();

    hidden_gradients = hidden_outputs & hidden_err;

    inputs.transpose();

    hidden_gradients = hidden_gradients * inputs;

    hidden_gradients = hidden_gradients * lr;

    hidden_weights = hidden_weights + hidden_gradients; 
}

void NeuralNetwork::save_weights(std::string path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return;
    }

    file << hidden_weights.getRows() << " " << hidden_weights.getColumns() << std::endl;
    for (int i = 0; i < hidden_weights.getRows(); i++) {
        for (int j = 0; j < hidden_weights.getColumns(); j++) {
            file << hidden_weights(i, j) << " ";
        }
        file << std::endl;
    }

    file << output_weights.getRows() << " " << output_weights.getColumns() << std::endl;
    for (int i = 0; i < output_weights.getRows(); i++) {
        for (int j = 0; j < output_weights.getColumns(); j++) {
            file << output_weights(i, j) << " ";
        }
        file << std::endl;
    }

    file.close();
}

void NeuralNetwork::load_weights(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return;
    }

    int weights_rows, weights_cols;
    Tensor* loaded_weights;

    for (int k = 0; k < 2; k++) {
        if (k == 0) {
            loaded_weights = &hidden_weights;
        } else {
            loaded_weights = &output_weights;
        }

        file >> weights_rows;
        file >> weights_cols;
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        if (weights_rows != (*loaded_weights).getRows() || weights_cols != (*loaded_weights).getColumns()) {
            std::cout << "Invalid shape of weights to load" << std::endl;
            return;
        }

        for (int i = 0; i < (*loaded_weights).getRows(); i++) {
            for (int j = 0; j < (*loaded_weights).getColumns(); j++) {
                file >> (*loaded_weights)(i, j);
            }
            file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
    }

    file.close();
}
