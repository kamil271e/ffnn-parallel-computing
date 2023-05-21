#include "../lib/tensor.hpp"
#include <random>

Tensor::Tensor(int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    values.resize(rows, std::vector<double>(columns));
}

void Tensor::normalDistInit(double mean=0.0, double std=1.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, std);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->columns; j++) {
            this->values[i][j] = distribution(gen);
        }
    }
}

void Tensor::display() {
    for (int i = 0; i < this->rows; i++) {
            std::cout << "[ ";
            for (int j = 0; j < this->columns; j++) {
                std::cout << this->values[i][j] << " ";
            } std::cout << "]" << std::endl;
        } 
}

void Tensor::transpose() {
}

void Tensor::flatten() {
}

int Tensor::argmax() {
    return -1;
}

void Tensor::relu() {
}

void Tensor::softmax() {
}