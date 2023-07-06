#pragma once
#include "tensor.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class Digit {
public:
    Digit();
    Digit(Tensor&, int);
    void display();
    Tensor data;  
    int label;
};