#include <iostream>
#include <chrono>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"
#include <omp.h>

int main(int argc, char** argv){ 
    int train_size;
    std::vector<Digit> train_set;
    int input_size = 28*28;
    int num_classes = 10;
    double lr = 0.01;
    bool parallel = true;
    omp_set_num_threads(NUM_THREADS);

    if (argc < 2){ // REGULAR TRAINING
        train_size = 1000;
        train_set = loadMNIST("datasets/mnist_train.csv", train_size, parallel);
        NeuralNetwork model(input_size, 500, num_classes, lr, parallel);
        auto start = std::chrono::high_resolution_clock::now();
        double accuracy = model.fit(train_set);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);        
        std::cout << duration.count() << " " << accuracy << std::endl;
    }
    else{ // TESTING FOR DIFFERENT TRAIN SIZES
        train_size = std::stoi(argv[1]);
        train_set = loadMNIST("datasets/mnist_train.csv", train_size, parallel);
        std::string path = "times/" + std::to_string(train_size) + ".txt";
        std::ofstream file(path);
        if (!file.is_open()) {
            std::cout << "Failed to open: " << path << std::endl;
            return -1;
        }

        for (int i = 2; i < argc; i++){
            int neurons = std::stoi(argv[i]);
            NeuralNetwork model(input_size, neurons, num_classes, lr);
            
            auto start = std::chrono::high_resolution_clock::now();
            double accuracy = model.fit(train_set);
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            
            file << neurons << " " << duration.count() << " " << accuracy << std::endl;
            // std::cout << train_size << " " << duration.count() << " " << accuracy << std::endl;
        }
        file.close();
    }
    
    return 0;
}