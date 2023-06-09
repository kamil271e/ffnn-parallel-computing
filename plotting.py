import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')
root_dir = 'times'


for train_size in os.listdir(root_dir):
    train_size_path = os.path.join(root_dir, train_size)

    # Time plot
    fig_time = plt.figure(figsize=(8, 6))
    ax_time = fig_time.add_subplot(111)
    
    # Accuracy plot
    fig_accuracy = plt.figure(figsize=(8, 6))
    ax_accuracy = fig_accuracy.add_subplot(111)

    thread_files = sorted([int(x.split('.')[0]) for x in os.listdir(train_size_path)])
    for file in thread_files:
        file = str(file) + '.txt'
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
            ax_time.plot(neurons, times, '-o', label=f'Num threads: {num_threads}')
            ax_time.set_title(f'Time of execution [train size: {train_size}]')
            ax_time.set_xlabel('No. of neurons in hidden layer')
            ax_time.set_ylabel('Time [s]')
            ax_time.legend()

            # Accuracy plot
            ax_accuracy.plot(neurons, accuracies, '-o', label=f'Num threads: {num_threads}')
            ax_accuracy.set_title(f'Accuracy [train size: {train_size}]')
            ax_accuracy.set_xlabel('No. of neurons in hidden layer')
            ax_accuracy.set_ylabel('Accuracy')
            ax_accuracy.legend()

    plt.tight_layout()

    fig_time.savefig(f'plots/time_{train_size}.svg')
    plt.close(fig_time)

    fig_accuracy.savefig(f'plots/accuracy_{train_size}.svg')
    plt.close(fig_accuracy)

print('Plots generated and saved.')
