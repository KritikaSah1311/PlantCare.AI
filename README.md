# PlantCare.AI 🌱

PlantCare.AI is a machine learning project that detects plant diseases from leaf images using deep learning.
The system helps farmers and gardeners quickly identify plant health issues and take corrective action.

## Features

* Plant disease detection using image classification
* Deep learning model based on MobileNetV2
* Fast prediction using transfer learning
* Can be extended into a mobile or web application

## Technologies Used

* Python
* TensorFlow / Keras
* MobileNetV2
* NumPy
* Matplotlib
* Google Colab

## Model

The model is trained using **MobileNetV2 transfer learning** for efficient image classification.

Saved model file:

```
mobilenetv2_best.keras
```

## Dataset

The dataset contains images of plant leaves belonging to different disease categories.
Images are preprocessed using TensorFlow's ImageDataGenerator for training and validation.

## How to Run the Project

1. Clone the repository

```
git clone https://github.com/KritikaSah1311/PlantCare.AI.git
```

2. Install dependencies

```
pip install tensorflow numpy matplotlib
```

3. Run the training or prediction script in Python or Google Colab.

## Future Improvements

* Add a web interface for farmers
* Deploy the model using a REST API
* Add more plant disease datasets
* Convert the system into a mobile app

## Author

Kritika Sah
Surya Bharadwaj
Vaishnavi Shrivastava
Keerty Narote
