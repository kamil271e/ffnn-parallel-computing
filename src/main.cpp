#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"

int main(){
    std::vector<Digit> train_set = loadMNIST("../datasets/mnist_train.csv", 500);
    // train_set[1].display();
    // std::cout << "Label: " << train_set[1].label << std::endl;
    Linear model(28*28, 100, 10, 0.01);
    model.fit(train_set);
    return 0;
}