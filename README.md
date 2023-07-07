# ffnn-parallel-computing
Feed forward neural network written in C++ leverages OpenMP to parallelize calculations, designed to classify hand written digits from MNIST dataset. Assignment for Parallel Processing classes at Poznan University of Technology.

### Results

The classification performance is influenced by the choice of hyperparameters, weight initialization, and the input data size. Both the sequential and parallel versions yield similar results. The table presents **accuracy values** for a test set with 500 samples, a learning rate of 0.01, 500 neurons in the hidden layer, and weight initialization using values drawn from a uniform distribution of $[-\frac{1}{\sqrt{n}}, \frac{1}{\sqrt{n}}]$ (where $n$ represents the number of rows).

| Train Size | Training  | Inference  |
|------------|------------------|--------------------|
| 200        | 0.62             | 0.629              |
| 500        | 0.724            | 0.762              |
| 1000       | 0.778            | 0.82               |
| 5000       | 0.877            | 0.914              |


![accuracy_1000](https://github.com/kamil271e/ffnn-parallel-computing/assets/82380348/3b24cb32-c061-4893-b8e2-639f922203bc) | ![accuracy_2000](https://github.com/kamil271e/ffnn-parallel-computing/assets/82380348/e6e70b27-11f6-4a30-b85c-13b23ad9014e)
:---------------------:|:---------------------:

### Execution times

As can be seen from the plots, the training times for the parallel processing solution are smaller compared to those observed during sequential learning - the best results are achieved when using 12 threads. The performance difference becomes more pronounced as the number of neurons in the hidden layer increases. This observation is expected since the network matrix dimensions have a direct relationship with the number of hidden neurons.

![time_1000](https://github.com/kamil271e/ffnn-parallel-computing/assets/82380348/decaec99-872c-44f1-9bf9-5591ca79f8cd) | ![time_2000](https://github.com/kamil271e/ffnn-parallel-computing/assets/82380348/6d08fb93-1da0-437b-94c3-1865ef0d6707)
:---------------------:|:---------------------:
