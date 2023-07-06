#include "../lib/digit.hpp"

Digit::Digit(){}

Digit::Digit(Tensor& data, int label)
    : data(data), label(label) {}


void Digit::display(){ // pretty printing
    // data.display();
    for (int i = 0; i < data.getRows(); i++){
        for (int j = 0; j < data.getColumns(); j++){
            std::cout.width(7);
            std::cout << std::right << data(i,j) << " ";
        } std::cout << std::endl << std::endl;
    }
}

Digit* loadMNIST(std::string path, int n_samples, bool parallel=false) {
    Digit* digits = new Digit[n_samples];
    Tensor img(28, 28, parallel);

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open: " << path << std::endl;
        return digits;
    }

    std::string row;
    std::getline(file, row);
    int i = 0;
    while (std::getline(file, row) && i < n_samples) {
        std::istringstream iss(row);
        std::string token;

        std::getline(iss, token, ',');
        int label = std::stoi(token);
       
        int j = 0;
        while (std::getline(iss, token, ',')) {
            double pixel = std::stod(token) / 255.0; // pixel normalization
            img(j/28, j%28) = pixel;
            j++;
        }
        Tensor imgCopy = img; // DeepCopy
        new (&digits[i]) Digit(imgCopy, label);
        i++;
    }

    file.close();
    return digits;
}