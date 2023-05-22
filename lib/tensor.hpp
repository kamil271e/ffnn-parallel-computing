#pragma once
#include <iostream>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(int, int);
    void setValue(int, int, double);
    double getValue(int, int);
    int getRows();
    int getColumns();
    
    void normalDistInit(double mean=0.0, double std=1.0);
    void display();
    void transpose();
    void flatten(int axis=0);
    int argmax(int axis=0);
    
    void relu();
    void sigmoid();
    void softmax();

    Tensor operator*(Tensor&);
    Tensor operator+(Tensor&);
    Tensor operator-(Tensor&);
private:
    int rows, columns;
    std::vector<std::vector<double>> values;
};   
