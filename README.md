# Emotion Detection

A Keras CNN trained on 48×48 grayscale faces, then a webcam loop that finds a face and writes one of seven emotion labels on the frame.

Labels: angry, disgusted, fear, happy, neutral, sad, surprised.

Live page: [maadhiniportfolio.vercel.app/projects/emotion-detection](https://maadhiniportfolio.vercel.app/projects/emotion-detection)

## Files

- `datatrain.ipynb` — builds the network and trains it
- `datatest.ipynb` — loads the saved model and runs the camera
- `emotion_model.json` — model architecture
- `emotion_model.h5` — trained weights
- `new_target.npy` — a saved target array from data prep

## Training

`ImageDataGenerator` rescales pixels by `1/255` and loads 48×48 grayscale batches of 64 from local train and test folders. The network is conv 32 and 64, pool, dropout 0.25, conv 128, pool, conv 128, pool, dropout 0.25, flatten, dense 1024, dropout 0.5, dense 7 softmax. It trains for 50 epochs with Adam at `0.0001`.

The image folders came from a Kaggle download and are not in this repository. The paths in the notebook are local Windows paths.

## Camera

`datatest.ipynb` rebuilds the model from the JSON and the h5 file. A frontal-face Haar cascade finds the face. The crop is resized to 48×48 and passed to `predict`. A green box and the winning label are drawn on the frame. Escape closes the window.
