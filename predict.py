import argparse
from tensorflow.keras.models import load_model
from utils import load_labels_csv, preprocess_image, decode_predictions

def predict(image_path, model_path, labels_path):
    # Load model
    model = load_model(model_path)

    # Load labels
    labels = load_labels_csv(labels_path)

    # Preprocess image
    image_dims = (640, 640, 3)
    image = preprocess_image(image_path, image_dims)

    # Make predictions
    predictions = model.predict(image)

    # Decode predictions
    mlb = MultiLabelBinarizer()
    mlb.fit(labels)
    decoded_labels = decode_predictions(predictions, mlb)

    # Print predictions
    print("Predicted labels:")
    for label in decoded_labels[0]:
        print(label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--labels", required=True, help="Path to the labels CSV file")
    args = parser.parse_args()

    predict(args.image, args.model, args.labels)
