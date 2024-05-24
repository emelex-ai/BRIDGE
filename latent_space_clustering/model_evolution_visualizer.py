import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

class ModelEvolutionVisualizer:
    def __init__(self, model_states, data):
        self.model_states = model_states
        self.data = data

    def animate_evolution(self):
        fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            labels = self.model_states[frame].predict(self.data)
            sns.scatterplot(x=self.data[:, 0], y=self.data[:, 1], hue=labels, palette='viridis', ax=ax)
            ax.set_title(f'Cluster Evolution at Epoch {frame}')

        ani = FuncAnimation(fig, update, frames=len(self.model_states), repeat=False)
        plt.show()

# Example usage
# model_evolution_visualizer = ModelEvolutionVisualizer(model_states, latent_data)
# model_evolution_visualizer.animate_evolution()
