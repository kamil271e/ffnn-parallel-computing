#include <iostream>
#include "tensor.cpp"

int main(){
    Tensor T(2,3);
    T.normalDistInit();
    T.display();
    std::cout<<std::endl;
    
    T.transpose();
    T.display();
    std::cout<<std::endl;
    
    T.flatten();
    T.display();

    int argmax = T.argmax();
    std::cout << argmax << std::endl <<std::endl;

    // T.relu();
    // T.display();

    T.sigmoid();
    T.display();
    std::cout<<std::endl;

    T.softmax();
    T.display();
    std::cout<<std::endl;

    return 0;
}