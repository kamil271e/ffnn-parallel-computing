import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
root_dir = 'times'

for train_size in os.listdir(root_dir):
    train_size_path = os.path.join(root_dir, train_size)

    for file in os.listdir(train_size_path):
        path = os.path.join(train_size_path, file)
        num_threads = file.split('.')[0]
        neurons, times, accuracies = [], [], []
        with open(path) as f:
            line = f.readline()
            while line:
                neuron, execution_time, accuracy = line.split(' ')
                neurons.append(int(neuron))
                times.append(float(execution_time))
                accuracies.append(float(accuracy))
                line = f.readline()
            
            # Time plot
            plt.plot(neurons, times, '-o', label=f'Num threads: {num_threads}')
            plt.title(f'Time of execution [train size: {train_size}]')
            plt.xlabel('No. of neurons in hidden layer')
            plt.ylabel('Time [s]')

    plt.legend()
    plt.savefig(f'plots/time_{train_size}.svg')
    plt.clf()

    # neurons, times, accuracies = [], [], []
    # train_size = int(file.split('.')[0])
   
    # with open(path) as f:
    #     line = f.readline()
    #     while line:
    #         neuron, execution_time, accuracy = line.split(' ')
    #         neurons.append(int(neuron))
    #         times.append(float(execution_time))
    #         accuracies.append(float(accuracy))
    #         line = f.readline()

    #     # Time plot
    #     plt.plot(neurons, times, '-o')
    #     plt.title(f'Time of execution [train size: {train_size}]')
    #     plt.xlabel('No. of neurons in hidden layer')
    #     plt.ylabel('Time [s]')
    #     plt.savefig(f'plots/time_{train_size}.svg')
    #     plt.clf()

    #     # Accuracy plot
    #     plt.plot(neurons, accuracies, '-o')
    #     plt.title(f'Train accuracy [train size: {train_size}]')
    #     plt.xlabel('No. of neurons in hidden layer')
    #     plt.ylabel('Accuracy')
    #     plt.savefig(f'plots/accuracy_{train_size}.svg')
    #     plt.clf()

print('Plots generated and saved.')
