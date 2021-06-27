import tensorflow as tf
from tensorflow.keras import layers, models
from numpy import asarray

def main():
    model = models.Sequential([
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
    ])

if __name__ == "__main__":
    main()
