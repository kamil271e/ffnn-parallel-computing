import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

for file in os.listdir('times'):
    path = os.path.join('times', file)
    neurons, times, accuracies = [], [], []
    train_size = int(file.split('.')[0])
   
    with open(path) as f:
        line = f.readline()
        while line:
            neuron, execution_time, accuracy = line.split(' ')
            neurons.append(int(neuron))
            times.append(float(execution_time) / 10**3)
            accuracies.append(float(accuracy))
            line = f.readline()

        # Time plot
        plt.plot(neurons, times, '-o')
        plt.title(f'Time of execution [train size: {train_size}]')
        plt.xlabel('No. of neurons in hidden layer')
        plt.ylabel('Time [s]')
        plt.savefig(f'plots/time_{train_size}.svg')
        plt.clf()

        # Accuracy plot
        plt.plot(neurons, accuracies, '-o')
        plt.title(f'Train accuracy [train size: {train_size}]')
        plt.xlabel('No. of neurons in hidden layer')
        plt.ylabel('Accuracy')
        plt.savefig(f'plots/accuracy_{train_size}.svg')
        plt.clf()

print('Plots generated and saved.')
