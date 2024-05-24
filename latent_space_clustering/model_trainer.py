from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, model, data, labels):
        self.model = model
        self.data = data
        self.labels = labels

    def train(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

# Example usage
# model_trainer = ModelTrainer(gmm, latent_data, labels)
# accuracy = model_trainer.train()
