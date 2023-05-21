#include "../lib/net.hpp"

Linear::Linear(int input_size, int hidden_size, int num_classes, double lr){
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->num_classes = num_classes;
    this->lr = lr;
    init_weights();
}

void Linear::init_weights(){
    hidden_layer = Tensor(hidden_size, input_size);
    hidden_layer.normalDistInit();

    output_layer = Tensor(num_classes, hidden_size);
    output_layer.normalDistInit();
}

void Linear::fit(std::vector<Digit> digits){
    for(const Digit& digit: digits){
        pass_forward(digit);
        break;
    }
}

void Linear::pass_forward(Digit digit){
    Tensor input = digit.data;
    input.flatten();

    Tensor hidden_output = hidden_layer * input;
    hidden_output.relu();

    Tensor output = output_layer * hidden_output;
    output.softmax();

    output.display();
}