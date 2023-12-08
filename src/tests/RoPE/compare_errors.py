import sys
import csv
import matplotlib.pyplot as plt
import re

# Function to extract loss from a log file and write to a CSV file
def extract_loss(file, csv_filename):
    with open(file, 'r') as f:
        lines = f.readlines()
        epochs = []
        losses = []
    # Regular expression pattern to match the loss and epoch
    pattern = r"metrics\[0\]:.*'train/loss': tensor\((\d+\.\d+).*'train/epoch': (\d+)"

    for line in lines:
        match = re.search(pattern, line)

        if match:
            losses.append(float(match.group(1)))
            epochs.append(int(match.group(2)))
    
    # Write the losses and epochs to a csv file
    with open(f'{csv_filename}', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'loss'])
        for i in range(len(epochs)):
            writer.writerow([epochs[i], losses[i]])

# Function to plot graph from a CSV file
def plot_graph(csv_filename, label, ax=None):
    epochs = []
    losses = []
    with open(csv_filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            if row[0] == 'epoch':
                continue
            epochs.append(int(row[0]))
            losses.append(float(row[1]))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.plot(epochs, losses, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    return ax

# Function to compare errors from two log files
def compare_errors(file1, file2, csv1, csv2):
    extract_loss(file1, csv1)
    extract_loss(file2, csv2)
    ax = plot_graph(csv1, label=f'{file1}')
    plot_graph(csv2, label=f'{file2}', ax=ax)
    plt.legend()
    plt.savefig(f'{csv1}_{csv2}.png')
    plt.show()

if __name__ == "__main__":
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    csv1 = file1 + '.csv'
    csv2 = file2 + '.csv'
    compare_errors(file1, file2, csv1, csv2)
