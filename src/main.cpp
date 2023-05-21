#include <iostream>
#include "tensor.cpp"

int main(){
    Tensor T(3,3);
    T.normalDistInit();
    T.display();
    return 0;
}