from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

BATCH_SIZE = 64
EPOCHS = 20


# Checking for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


class TrainingLoop:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Train mode
        self.model.train()

        # Predict
        y_hat = self.model(x)

        # Compute loss
        loss = self.loss_fn(y, y_hat)

        # Compute gradients
        loss.backward()

        # Update
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def train(self, train, val, batch_size=BATCH_SIZE, n_epochs=EPOCHS, n_features=1):
        model_path = f'models/{self.model.__class__.__name__ }_{datetime.now().strftime("%Y%m%d%H%M%S")}.pt'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    y_hat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, y_hat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch} / {n_epochs}] Training Loss: {training_loss:.4f}\t Validation Loss: {validation_loss:.4f}"
                )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test, batch_size=1, n_features=1):
        with torch.no_grad():
            predictions = []
            values = []

            for x_test, y_test in test:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                y_hat = self.model(x_test)
                predictions.append(y_hat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_loss(self):
        fig = plt.figure(figsize=(25, 12))
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Training & Validation Losses")
        plt.show()
        plt.close()
