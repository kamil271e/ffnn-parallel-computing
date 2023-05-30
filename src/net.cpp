#include "../lib/net.hpp"

Linear::Linear(int input_size, int hidden_size, int num_classes, double lr){
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->num_classes = num_classes;
    this->lr = lr;
    this->accurate_pred = 0;
    init_weights();
}

void Linear::init_weights(){
    hidden_weights = Tensor(hidden_size, input_size);
    hidden_weights.initUniform();

    output_weights = Tensor(num_classes, hidden_size);
    output_weights.initNorm();
}

void Linear::fit(std::vector<Digit> digits){
    for(const Digit& digit: digits){
        forward_propagation(digit);
        backward_propagation(digit.label);
    }
    std::cout << "Train accuracy: " << (double)accurate_pred  / (double)digits.size() << std::endl;
}

void Linear::predict(std::vector<Digit> digits){
    accurate_pred = 0;
    for(const Digit& digit: digits){
        forward_propagation(digit);
    }
    std::cout << "Test accuracy: " << (double)accurate_pred  / (double)digits.size() << std::endl;
}

void Linear::forward_propagation(Digit digit){
    inputs = digit.data;
    inputs.flatten();

    hidden_outputs = hidden_weights * inputs;
    hidden_outputs.relu();

    outputs = output_weights * hidden_outputs;
    outputs.softmax();

    if (digit.label == outputs.argmax()) accurate_pred++;
}

void Linear::backward_propagation(int target){
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

void Linear::save_weights(std::string path){
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return;
    }

    file << hidden_size << " " << input_size << std::endl;
    for (int i  = 0; i < hidden_size; i++){
        for (int j = 0; j < input_size; j++){
            file << hidden_weights(i,j) << " ";
        } file << std::endl;
    }
    
    file << num_classes << " " << hidden_size << std::endl;
    for (int i = 0; i < num_classes; i++){
        for (int j = 0; j < hidden_size; j++){
            file << output_weights(i,j) << " ";
        } file << std::endl;
    }

    file.close();
}

void Linear::load_weights(std::string path){
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return;
    }
}