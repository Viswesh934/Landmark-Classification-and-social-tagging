# Landmark Classification & Tagging for Social Media 2.0

This project implements a landmark classification system using deep learning techniques. It includes both a CNN model built from scratch and a transfer learning approach, along with a simple app for making predictions.

## Project Structure

- `cnn_from_scratch.ipynb`: CNN from scratch implementation
- `transfer_learning.ipynb`: Transfer learning approach
- `app.ipynb`: Simple prediction application
- `src/`:
  - `data.py`: Data loading and preprocessing
  - `model.py`: CNN architecture
  - `helpers.py`: Utility functions
  - `predictor.py`: Prediction logic
  - `transfer.py`: Transfer learning implementation
  - `optimization.py`: Loss and optimizer definitions
  - `train.py`: Training and validation functions

## Key Features

1. **Data Preprocessing**: Implemented resizing, cropping, normalization, and data augmentation.
2. **CNN from Scratch**: Custom CNN architecture for landmark classification.
3. **Transfer Learning**: Fine-tuned pre-trained model for improved performance.
4. **Training and Validation**: Implemented training loops with learning rate scheduling.
5. **Model Export**: Used TorchScript for model serialization.
6. **Simple App**: Basic application for model predictions on new images.

## Results

- CNN from scratch: >50% test accuracy
- Transfer learning: >60% test accuracy

## Usage

1. Run `cnn_from_scratch.ipynb`
2. Run `transfer_learning.ipynb`
3. Use `app.ipynb` to test the model on new images

## Future Work

- Experiment with advanced architectures and ensemble methods
- Implement more extensive data augmentation
- Explore few-shot learning for new landmark classes
