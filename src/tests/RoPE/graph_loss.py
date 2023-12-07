#use csv file passed in from command line to plot graph
import matplotlib.pyplot as plt
import csv
import sys

def plot_graph(csv_filename):
    epochs = []
    losses = []
    with open(csv_filename,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            epochs.append(int(row[0]))
            losses.append(float(row[1]))
    plt.plot(epochs,losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    csv_filename = sys.argv[1]
    plot_graph(csv_filename)