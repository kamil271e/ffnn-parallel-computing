#pragma once
#include "tensor.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class Digit {
public:
    Digit(Tensor&, int);
    void display();
    
    int label;
    Tensor data;
};