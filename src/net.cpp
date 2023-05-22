#include "../lib/net.hpp"

Linear::Linear(int input_size, int hidden_size, int num_classes, double lr){
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->num_classes = num_classes;
    this->lr = lr;
    this->accurate_pred = 0;
    init_weights();
}

void Linear::init_weights(){
    hidden_weights = Tensor(hidden_size, input_size);
    hidden_weights.normalDistInit();

    output_weights = Tensor(num_classes, hidden_size);
    output_weights.normalDistInit();
}

void Linear::fit(std::vector<Digit> digits){
    for(const Digit& digit: digits){
        pass_forward(digit);
        break;
    }
    // std::cout << "Accuracy: " << (double)accurate_pred / (double)digits.size() << std::endl;
}

void Linear::pass_forward(Digit digit){
    Tensor input = digit.data;
    input.flatten();

    Tensor hidden_output = hidden_weights * input;
    hidden_output.relu();

    Tensor output = output_weights * hidden_output;
    output.softmax();
    output.display();
    std::cout<<std::endl;

    Tensor labels(10,1);
    labels.oneHotEncoding(8);
    labels.display();
    std::cout<<std::endl;

    output.crossEntropyError(labels);
    Tensor gradTensor = output;
    gradTensor.display();
    std::cout<<std::endl;
    
    int predicted = output.argmax();
    if (digit.label == predicted){
        accurate_pred++;
    }
}