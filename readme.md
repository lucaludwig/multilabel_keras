# Multi-Label Image Classification with SmallerVGGNet

This project demonstrates multi-label image classification using the SmallerVGGNet architecture. It trains a convolutional neural network to classify images into multiple categories simultaneously.

## Installation

1. Clone the repository:

git clone https://github.com/lucaludwig/multilabel_keras.git

2. Install the required dependencies:

pip install -r requirements.txt

## Usage

1. Prepare the dataset:
- Place your image dataset in the `dataset` directory.
- Update the paths and filenames in the code as necessary.

2. Run the training script:

python train.py


3. Monitor the training progress and evaluation metrics.

4. After training, the model and label binarizer will be saved to disk.

5. Use the trained model for predictions:
- Update the path to the test image in the code.
- Run the prediction script:

python predict.py


## Project Structure

- `train.py`: Script to train the model on the dataset.
- `predict.py`: Script to make predictions using the trained model.
- `model.py`: Contains the implementation of the SmallerVGGNet model.
- `utils.py`: Utility functions for data preprocessing and evaluation.
- `dataset/`: Directory to store the image dataset.
- `model.h5`: Saved trained model.
- `mlb`: Saved MultiLabelBinarizer.

## Results

The training script will generate training loss, validation loss, training accuracy, and validation accuracy plots. These plots can help visualize the model's performance and assist in model selection.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
