#pragma once
#include <iostream>
#include <vector>
#include <math.h>

class Tensor {
public:
    Tensor();
    Tensor(int, int, bool parallel=false);
    Tensor(Tensor&);
    ~Tensor();
    void allocate();
    void deallocate();
    int getRows();
    int getColumns();
    
    void initNorm(double mean=0.0, double std=0.1);
    void initUniform();
    void ones();
    void display();
    void transpose();
    void flatten(int axis=0);
    int argmax(int axis=0);
    double maxval(int axis=0);
    double minval(int axis=0);
    
    void relu();
    void reluDerivative();
    void sigmoid();
    void sigmoidDerivative();
    void softmax();
    void softmaxDerivative();
    void oneHotEncoding(int);
    void crossEntropyError(Tensor);

    Tensor operator*(double);
    Tensor operator*(Tensor&);
    Tensor operator&(Tensor&);
    Tensor operator+(Tensor&);
    Tensor operator-(Tensor&);
    Tensor operator/(Tensor&);
    double& operator()(int, int);
private:
    int rows, columns;
    bool parallel;
    double** values;
};   
