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
    hidden_weights.initNorm();

    output_weights = Tensor(num_classes, hidden_size);
    output_weights.initNorm();
}

void Linear::fit(std::vector<Digit> digits){
    for(const Digit& digit: digits){
        fit_batch(digit);
        //break;
    }
    std::cout << "Accuracy: " << (double)accurate_pred / (double)digits.size() << std::endl;
}

void Linear::fit_batch(Digit digit){
    
    // FORWARD PROPAGATION
    Tensor inputs = digit.data;
    inputs.flatten();

    Tensor hidden_outputs = hidden_weights * inputs;
    hidden_outputs.relu();

    Tensor outputs = output_weights * hidden_outputs;
    outputs.softmax();
    
    if (digit.label == outputs.argmax()) accurate_pred++;


    // BACKWARD PROPAGATION
    Tensor labels(num_classes,1);
    labels.oneHotEncoding(digit.label);

    outputs.crossEntropyError(labels);
    
    Tensor gradients = outputs;
}
