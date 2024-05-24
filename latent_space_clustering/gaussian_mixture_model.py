from sklearn.mixture import GaussianMixture

class GaussianMixtureModel:
    def __init__(self, n_components):
        self.n_components = n_components
        self.gmm = GaussianMixture(n_components=self.n_components)

    def train(self, data):
        self.gmm.fit(data)

    def predict(self, data):
        return self.gmm.predict(data)

    def get_params(self):
        return self.gmm.get_params()

# Example usage
# gmm = GaussianMixtureModel(n_components=3)
# gmm.train(latent_data)
# labels = gmm.predict(latent_data)
