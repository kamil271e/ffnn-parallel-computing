#pragma once
#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "digit.hpp"

class NeuralNetwork {
public:
    NeuralNetwork(int, int, int, double, bool parallel=false);
    void init_weights();
    double fit(Digit*, int);
    void forward_propagation(Digit);
    void backward_propagation(int);
    double predict(Digit*, int);
    void save_weights(std::string);
    void load_weights(std::string);
private:
    int input_size, hidden_size, num_classes;
    Tensor inputs;
    Tensor hidden_weights;
    Tensor hidden_outputs;
    Tensor output_weights;
    Tensor outputs;
    Tensor hidden_err;
    Tensor output_err;
    Tensor hidden_gradients;
    Tensor output_gradients;
    Tensor labels;
    double lr;
    int accurate_pred;
    bool parallel;
};
