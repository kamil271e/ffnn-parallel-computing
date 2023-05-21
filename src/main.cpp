#include <iostream>
#include "tensor.cpp"

int main(){
    Tensor T(1,2);
    T.normalDistInit();
    T.display();
    std::cout<<std::endl;

    // Tensor V(2,2);
    // V.normalDistInit();
    // V.display();
    // std::cout<<std::endl;

    // Tensor X = T * V;
    // X.display();
    // std::cout<<std::endl;

    Tensor Z(1,2);
    Z.normalDistInit();
    Z.display();
    std::cout<<std::endl;

    Tensor F = T + Z;
    F.display();
    std::cout<<std::endl;

    Tensor U = F - T;
    U.display();

    return 0;
}