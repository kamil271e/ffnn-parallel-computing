#include "../lib/digit.hpp"


Digit::Digit(Tensor& data, int label)
    : data(data), label(label) {}


void Digit::display(){ // pretty printing
    // data.display();
    for (int i = 0; i < data.getRows(); i++){
        for (int j = 0; j < data.getColumns(); j++){
            std::cout.width(7);
            std::cout << std::right << data.getValue(i,j) << " ";
        } std::cout << std::endl << std::endl;
    }
}

std::vector<Digit> loadMNIST(std::string path, int n_samples) {
    std::vector<Digit> digits;
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

        Tensor img = Tensor(28, 28);
        int j = 0;
        while (std::getline(iss, token, ',')) {
            double pixel = std::stod(token) / 255.0; // pixel normalization
            img.setValue(j / 28, j % 28, pixel);
            j++;
        }
        Digit digit(img, label);
        digits.push_back(digit);
        i++;
    }

    file.close();
    return digits;
}