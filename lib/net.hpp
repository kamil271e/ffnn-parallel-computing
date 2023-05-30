#pragma once
#include <iostream>
#include <vector>
#include "tensor.hpp"
#include "digit.hpp"

class Linear {
public:
    Linear(int, int, int, double);
    void fit(std::vector<Digit>);
    void fit_batch(Digit);
    void init_weights();
    void predict(std::vector<Digit>);
private:
    int input_size, hidden_size, num_classes;
    Tensor hidden_weights;
    Tensor output_weights;
    double lr;
    int accurate_pred;
};
