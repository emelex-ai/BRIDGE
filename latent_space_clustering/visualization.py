import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def plot_clusters(self):
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=self.data[:, 0], y=self.data[:, 1], hue=self.labels, palette='viridis')
        plt.title('Cluster Visualization')
        plt.show()

# Example usage
# visualization = Visualization(latent_data, labels)
# visualization.plot_clusters()
