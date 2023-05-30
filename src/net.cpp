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
    hidden_weights.initUniform();

    output_weights = Tensor(num_classes, hidden_size);
    output_weights.initUniform();
    output_weights.initNorm();
}

void Linear::fit(std::vector<Digit> digits){
    int  i = 0;
    for(const Digit& digit: digits){
        fit_batch(digit);
        i++;
    }
    std::cout << "Train accuracy: " << (double)accurate_pred  / i << std::endl;
}

void Linear::predict(std::vector<Digit> digits){
    accurate_pred = 0;
    int i = 0;
    for(const Digit& digit: digits){
        fit_batch(digit);
        i++;
    }
    std::cout << "Test accuracy: " << (double)accurate_pred << " " <<  i << 
    " " << (double)accurate_pred / i << std::endl;
}

void Linear::fit_batch(Digit digit){
    
    // FORWARD PROPAGATION
    Tensor inputs = digit.data;
    inputs.flatten();

    Tensor hidden_outputs = hidden_weights * inputs;

    hidden_outputs.sigmoid();

    Tensor outputs = output_weights * hidden_outputs;
    
    outputs.sigmoid();
    
    if (digit.label == outputs.argmax()) accurate_pred++;

    int predicted = outputs.argmax();
    
    // BACKWARD PROPAGATION
    Tensor labels(num_classes, 1);
    labels.oneHotEncoding(digit.label);

    Tensor final_outputs = outputs;

    Tensor output_err = labels - final_outputs;
    outputs.crossEntropyError(labels);
    
    output_weights.transpose();

    Tensor hidden_err = output_weights * output_err;

    final_outputs.softmaxDerivative();

    Tensor output_gradients = final_outputs & output_err;

    hidden_outputs.transpose();

    output_gradients = output_gradients * hidden_outputs;

    output_gradients = output_gradients * lr;

    output_weights.transpose();

    output_weights = output_weights + output_gradients;

    ///////////////////////////////////////////////////

    hidden_outputs.sigmoidDerivative();

    hidden_outputs.transpose();

    Tensor hidden_gradients = hidden_outputs & hidden_err;

    inputs.transpose();

    hidden_gradients = hidden_gradients * inputs;

    hidden_gradients = hidden_gradients * lr;

    hidden_weights = hidden_weights + hidden_gradients; 

}
