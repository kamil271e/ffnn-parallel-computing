#pragma once
#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "digit.hpp"

class Linear {
public:
    Linear(int, int, int, double);
    void init_weights();
    void fit(std::vector<Digit>);
    void forward_propagation(Digit);
    void backward_propagation(int);
    void predict(std::vector<Digit>);
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
};
