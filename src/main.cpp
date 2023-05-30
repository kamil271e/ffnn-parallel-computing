#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"

int main(){
    std::vector<Digit> train_set = loadMNIST("../datasets/mnist_train.csv", 500);
    std::vector<Digit> test_set = loadMNIST("../datasets/mnist_test.csv", 100);

    Linear model(28*28, 1000, 10, 0.01);
    model.fit(train_set);
    model.predict(test_set);

    return 0;
}