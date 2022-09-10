# Abstract

A Deep Learning model that combines the concept of Stop-Loss with the capabilities offered by Deep Neural Networks. The architecture is composed of three components where trend detection and price prediction components provide inputs to the stop price prediction component which predicts the variation of stop price. 

# Requirements

* Python 3.8
* Tensorflow 2

# Install required packages

`pip install -r requirements.txt`

# Running the scripts

## Training

To train the model, run the Model/stop_loss_train.py with the following parameters

* $train_dataset - Training dataset
* $output_folder - Output directory (where the trained models will be saved to)

### Running the train script

`python Model/stop_loss_train.py $train_dataset $output_folder`

## Testing

To test the model, run the Model/stop_loss_test.py with the following parameters

* $test_dataset - Testing dataset
* $output_folder - Output directory (where the results will be saved to)
* $model - Path to the trained model
* $scaler - Path to the scaler used for training the model

### Running the test script

`python Model/stop_loss_testpy $test_dataset $output_folder $model $scaler`
