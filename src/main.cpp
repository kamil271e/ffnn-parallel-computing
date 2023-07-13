#include <iostream>
#include "tensor.cpp"
#include "digit.cpp"
#include "net.cpp"
#include <omp.h>

int main(int argc, char** argv){ 
    int train_size, threads_num, input_size, num_classes, neurons;
    bool parallel;
    double lr, start, stop, duration, accuracy; 
    Digit* train_set;

    input_size = 28*28;
    num_classes = 10;
    lr = 0.01;

    train_size = std::stoi(argv[1]);
    threads_num = std::stoi(argv[2]);
    omp_set_num_threads(threads_num);

    if (threads_num > 1){
        parallel = true;
    } else{
        parallel = false;
    }
    train_set = new Digit[train_size];
    train_set = loadMNIST("datasets/mnist_train.csv", train_size, parallel);

    if (argc < 5){ // REGULAR TRAINING
        neurons = std::stoi(argv[3]);
        NeuralNetwork model(input_size, neurons, num_classes, lr, parallel);
        start = omp_get_wtime();
        accuracy = model.fit(train_set, train_size);
        std::cout << "after fit" << std::endl;
        stop = omp_get_wtime();
        
        duration = stop - start;       
        std::cout << "Execution time: " << duration << " ; Accuracy: " << accuracy << std::endl;
    
        int test_size = 200;
        Digit* test_set = new Digit[test_size];
        test_set = loadMNIST("datasets/mnist_test.csv", test_size, parallel);

        accuracy = model.predict(test_set, test_size);
        std::cout << accuracy << std::endl;
    
    }
    else{ // TESTING FOR DIFFERENT TRAIN SIZES
        std::string path = "times/" + std::to_string(train_size) + "/" + std::to_string(threads_num) + ".txt";
        std::ofstream file(path);
        
        if (!file.is_open()) {
            std::cout << "Failed to open: " << path << std::endl;
            return -1;
        }

        for (int i = 3; i < argc; i++){
            neurons = std::stoi(argv[i]);
            NeuralNetwork model(input_size, neurons, num_classes, lr, parallel);
            
            start = omp_get_wtime();
            accuracy = model.fit(train_set, train_size);
            stop = omp_get_wtime();
            duration = stop - start;    

            file << neurons << " " << duration << " " << accuracy << std::endl;
            
            // for debug
            // std::cout << "[train_size: " << train_size <<"; neurons: " << neurons << "] Execution time: " << duration << " ; Accuracy: " << accuracy << " ; Threads: " << threads_num << std::endl;
        }
        file.close();
    }
    
    delete[] train_set;
    return 0;
}