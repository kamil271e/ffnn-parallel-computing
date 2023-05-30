#include "../lib/tensor.hpp"
#include <random>

Tensor::Tensor(){}

Tensor::Tensor(int rows, int columns) {
    this->rows = rows;
    this->columns = columns;
    values.resize(rows, std::vector<double>(columns));
}

void Tensor::setValue(int i, int j, double val){
    values[i][j] = val;
}

double Tensor::getValue(int i, int j){
    return values[i][j];
}

int Tensor::getColumns(){
    return columns;
}

int Tensor::getRows(){
    return rows;
}

// Initialize Tensor values with samples from normal distribution
void Tensor::initNorm(double mean, double std) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, std);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            values[i][j] = distribution(gen);
        }
    }
}

// Initialize tensor values with samples from uniform distribution
void Tensor::initUniform(){
    double min = -1.0 / sqrt(rows);
	double max = 1.0 / sqrt(rows);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(min, max);
    for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			values[i][j] = distribution(gen);
		}
	}
}

// Fill tensor with 1s
void Tensor::ones(){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            values[i][j] = 1.0;
        }
    }
}

void Tensor::display() {
    for (int i = 0; i < rows; i++) {
            std::cout << "[ ";
            for (int j = 0; j < columns; j++) {
                std::cout << values[i][j] << " ";
            } std::cout << "]" << std::endl;
        } 
}

void Tensor::transpose() {
    std::vector<std::vector<double>> transposed(columns, std::vector<double>(rows));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            transposed[j][i] = values[i][j];
        }
    }
    std::swap(rows, columns);
    values = std::move(transposed);
}

void Tensor::flatten(int axis) {
    // numpy based: axis=0 -> column vector
    // axis=1 -> row vector
    int n = rows * columns;
    std::vector<std::vector<double>> flattened;
    if (axis==0){
        flattened.resize(n, std::vector<double>(1));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                flattened[i * columns + j][0] = values[i][j];
            }
        }
        rows=n;
        columns=1;
    } else if(axis==1){
        flattened.resize(1, std::vector<double>(n));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                flattened[0][i * columns + j] = values[i][j];
            }
        }
        rows=1;
        columns=n;
    }else{
        std::cout << "Flatten operation failed. Axis parameter should be 0 or 1." << std::endl;
        return;
    }
    values = std::move(flattened);
}

int Tensor::argmax(int axis) {
    int cur_argmax[2] = {-1,-1};
    double cur_max = -10000000;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            if (values[i][j] > cur_max){
                cur_argmax[0] = i;
                cur_argmax[1] = j;
                cur_max = values[i][j];
            }
        }
    }
    if(axis != 0 && axis != 1){
        std::cout << "Argmax operation failed. Axis parameter should be 0 or 1." << std::endl;
        return -1;
    }
    return cur_argmax[axis];
}

double Tensor::maxval(int axis) {
    double cur_max = -10000000;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            if (values[i][j] > cur_max){
                cur_max = values[i][j];
            }
        }
    }
    if(axis != 0 && axis != 1){
        std::cout << "Maxval operation failed. Axis parameter should be 0 or 1." << std::endl;
        return -1;
    }
    return cur_max;
}

double Tensor::minval(int axis) {
    double cur_min = 10000000;
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            if (values[i][j] < cur_min){
                cur_min = values[i][j];
            }
        }
    }
    if(axis != 0 && axis != 1){
        std::cout << "Minval operation failed. Axis parameter should be 0 or 1." << std::endl;
        return -1;
    }
    return cur_min;
}

void Tensor::relu() {
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            values[i][j] = (values[i][j] > 0) ? values[i][j] : 0; 
        }
    }
}

void Tensor::reluDerivative() {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            values[i][j] = (values[i][j] <= 0.0) ? 0.0 : 1.0;
        }
    }
}

void Tensor::sigmoid(){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            values[i][j] = 1.0 / (1.0 + std::exp(-values[i][j]));
        }
    }
}

// void Tensor::sigmoidDerivative(){ // (1 - sigm(x)) * sigm(x) for each x
//     Tensor all_ones(rows, columns);
//     all_ones.ones();

//     Tensor sigmoid_output = *this;
//     sigmoid_output.sigmoid();

//     Tensor substracted(rows, columns);
//     substracted = all_ones - sigmoid_output;

//     *this = sigmoid_output & substracted;
// }

void Tensor::sigmoidDerivative(){ // (1 - sigm(x)) * sigm(x) for each x
    Tensor all_ones(rows, columns);
    all_ones.ones();

    Tensor substracted(rows, columns);
    substracted = all_ones - (*this);

    *this = *this & substracted;
}

void Tensor::softmax(){
    std::vector<std::vector<double>> probas(rows, std::vector<double>(columns));
    double sumExp = 0.0;

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            double expV = std::exp(values[i][j]);
            probas[i][j] = expV;
            sumExp += expV;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            probas[i][j] /= sumExp;
        }
    }
    values = std::move(probas);
}

void Tensor::softmaxDerivative() {
    sigmoidDerivative();
}

void Tensor::oneHotEncoding(int label){
    if (columns > 1 || label > rows-1){ // applied to column vector
        std::cout << "One hot encoding faied." << std::endl;
        return;
    }
    values[label][0] = 1.0;
}

void Tensor::crossEntropyError(Tensor labels){
    std::vector<std::vector<double>> err(rows, std::vector<double>(columns));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double label = labels.getValue(i, j);
            double prediction = values[i][j];
            const double epsilon = 1e-7; // in case prediciton = 0
            prediction = std::max(epsilon, prediction); 
            err[i][j] = -label * std::log(prediction);
        }
    }
    values = std::move(err);
}

Tensor Tensor::operator*(double scalar){ // SCALING
    Tensor finalTensor(rows, columns);
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            finalTensor.setValue(i, j, values[i][j] * scalar);
        }
    }
    return finalTensor;
}

Tensor Tensor::operator*(Tensor& T){ // DOT PRODUCT
    if (columns != T.getRows()){
        std::cout << "Dot operation failed. Incompatible dimensions" << std::endl;
        return Tensor();
    }
    int finalRows = rows;
    int finalColumns = T.getColumns();
    Tensor finalTensor(finalRows, finalColumns);

    for (int i = 0; i < finalRows; i++) {
        for (int j = 0; j < finalColumns; j++) {
            double sum = 0.0;
            for (int k = 0; k < columns; k++) {
                sum += values[i][k] * T.getValue(k,j);
            }
            finalTensor.setValue(i,j,sum);
        }
    }
    return finalTensor;
}

Tensor Tensor::operator&(Tensor& T){// ELEMENT-WISE MULTIPLICATION
    if (columns != T.getColumns() || rows != T.getRows()){
        std::cout << "Element-wise multiplication failed. Incompatible dimensions" <<std::endl;
        return Tensor();
    }
    Tensor finalTensor(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double product =  values[i][j] * T.getValue(i, j);
            finalTensor.setValue(i, j, product);
        }
    }
    return finalTensor;
}

Tensor Tensor::operator+(Tensor& T){ // ADD
    if (columns != T.getColumns() || rows != T.getRows()){
        std::cout << "Add operation failed. Incompatible dimensions" << std::endl;
        return Tensor();
    }
    Tensor finalTensor(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double add_result = values[i][j] + T.getValue(i,j);
            finalTensor.setValue(i, j, add_result);
        } 
    }
    return finalTensor;
}

Tensor Tensor::operator-(Tensor& T){ // SUBSTRACT
    if (columns != T.getColumns() || rows != T.getRows()){
        std::cout << "Substract operation failed. Incompatible dimensions" << std::endl;
        return Tensor();
    }
    Tensor finalTensor(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            double substract_result = values[i][j] - T.getValue(i,j);
            finalTensor.setValue(i, j, substract_result);
        }
    }
    return finalTensor;
}