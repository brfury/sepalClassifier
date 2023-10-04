import tensorflow as tf
import pandas as pd
import os

class IrisClassifier:
    def __init__(self):
        self.CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        self.SPECIES = ['Setosa', 'Versicolor', 'Virginica']
        self.train_path = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
        self.test_path = "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
        self.train = None
        self.test = None
        self.train_y = None
        self.test_y = None
        self.model = None

    def clear_terminal(self):
        os.system('cls')

    def load_data(self):
        self.train = pd.read_csv(self.train_path, names=self.CSV_COLUMN_NAMES, header=0)
        self.test = pd.read_csv(self.test_path, names=self.CSV_COLUMN_NAMES, header=0)
        self.train_y = self.train.pop('Species')
        self.test_y = self.test.pop('Species')

    def input_fn(self, features, labels, training=True, batch_size=256):
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        if training:
            dataset = dataset.shuffle(1000).repeat()
            self.clear_terminal()
        return dataset.batch(batch_size)

    def create_feature_columns(self):
        my_feature_columns = []
        for key in self.train.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        return my_feature_columns

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.DenseFeatures(self.create_feature_columns()),
            tf.keras.layers.Dense(30, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        self.model.fit(self.input_fn(self.train, self.train_y, training=True),
                       steps_per_epoch=5000 // 256,
                       epochs=5)

    def predict_species(self, SepalLength, SepalWidth, PetalLength, PetalWidth):
        new_data = {
            'SepalLength': [SepalLength],
            'SepalWidth': [SepalWidth],
            'PetalLength': [PetalLength],
            'PetalWidth': [PetalWidth]
        }
        self.clear_terminal()
        predictions = self.model.predict(self.input_fn(new_data, None, training=False))
        predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
        species = self.SPECIES[predicted_class_index]
        probability_percentage = predictions[0][predicted_class_index] * 100
        print(f"Prediction: {species} with {probability_percentage:.2f}% probability")

if __name__ == "__main__":
    iris_classifier = IrisClassifier()
    iris_classifier.load_data()
    iris_classifier.create_model()

    # local dos inputs
    SepalLength = float(input('Sepal Length\n'))
    SepalWidth = float(input('Sepal Width\n'))
    PetalLength = float(input('Petal Length\n'))
    PetalWidth = float(input('Petal Width\n'))

    iris_classifier.predict_species(SepalLength, SepalWidth, PetalLength, PetalWidth)
