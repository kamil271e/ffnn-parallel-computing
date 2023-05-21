#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"

int main(){
    std::vector<Digit> train_set = loadMNIST("../datasets/mnist_train.csv", 500);
    train_set[1].display();
    std::cout << "Label: " << train_set[1].label << std::endl;
    return 0;
}