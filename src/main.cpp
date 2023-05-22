#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"

int main(){
    // std::vector<Digit> train_set = loadMNIST("../datasets/mnist_train.csv", 1000);
    // train_set[1].display();
    // std::cout << "Label: " << train_set[1].label << std::endl;
    // Linear model(28*28, 500, 10, 0.01);
    // model.fit(train_set);
    Tensor T(2,2);
    T.ones();
    T.sigmoid();
    T.display();
    std::cout<<std::endl;

    T.softmaxDerivative();
    T.display();
    return 0;
}