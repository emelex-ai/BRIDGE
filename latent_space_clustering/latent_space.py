class LatentSpace:
    def __init__(self, data):
        self.data = data

    def get_latent_space(self):
        # For simplicity, assume data is already in latent space
        return self.data

    def manipulate_data(self, transform_function):
        self.data = transform_function(self.data)

# Example usage
# latent_space = LatentSpace(preprocessed_data)
# latent_data = latent_space.get_latent_space()
