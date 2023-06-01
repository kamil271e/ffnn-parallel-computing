#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"

int main(){
    std::vector<Digit> train_set = loadMNIST("../datasets/mnist_train.csv", 1000);
    std::vector<Digit> test_set = loadMNIST("../datasets/mnist_test.csv", 100);

    NeuralNetwork model(28*28, 100, 10, 0.03);
    // model.fit(train_set);
    
    model.load_weights("weights_.txt");
    model.predict(test_set);

    // model.save_weights("weights_.txt");
    return 0;
}