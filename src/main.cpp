#include <iostream>
#include <chrono>
 #include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"

int main(int argc, char** argv){
    int train_size;
    std::vector<Digit> train_set;

    if (argc < 2){ // REGULAR TRAINING
        train_size = 500;
        train_set = loadMNIST("datasets/mnist_train.csv", train_size);
        NeuralNetwork model(28*28, 400, 10, 0.03);
        double accuracy = model.fit(train_set);
        std::cout << "Train accuracy: " << accuracy << std::endl;
    }
    else{ // TESTING FOR DIFFERENT TRAIN SIZES
        train_size = std::stoi(argv[1]);
        train_set = loadMNIST("datasets/mnist_train.csv", train_size);
        std::string path = "times/" + std::to_string(train_size) + ".txt";
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cout << "Failed to open: " << path << std::endl;
            return -1;
        }

        for (int i = 2; i < argc; i++){
            int neurons = std::stoi(argv[i]);
            NeuralNetwork model(28*28, neurons, 10, 0.03);
            auto start = std::chrono::high_resolution_clock::now();
            double accuracy = model.fit(train_set);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << train_size << " " << duration.count() << " " << accuracy << std::endl;
            // store num of neurons, duration time and accuracy in file
            file << neurons << " " << duration.count() << " " << accuracy << std::endl;
        }
        file.close();
    }
    
    return 0;
}