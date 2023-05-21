#pragma once
#include <iostream>
#include <vector>

class Tensor {
public:
    Tensor(int, int);
    void setValue(int, int, double);
    double getValue(int, int);
    int getRows();
    int getColumns();
    
    void normalDistInit(double, double);
    void display();
    void transpose();
    void flatten(int);
    int argmax(int);
    
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
