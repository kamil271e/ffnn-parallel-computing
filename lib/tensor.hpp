#pragma once

#include <iostream>
#include <vector>

class Tensor {
public:
    Tensor(int, int);
    void normalDistInit(double, double);
    void display();
    void transpose();
    void flatten();
    int argmax();
    void relu();
    void softmax();
private:
    int rows, columns;
    std::vector<std::vector<double>> values;
};   
