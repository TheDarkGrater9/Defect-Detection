this folder contains all the necessary files for inference using the fine-tuned model obtained after running iterationtwo.ipynb, and versions of that model exported to ONNX.

sampleinference.ipynb : Performs dataset preprocessing and inference on the entire test dataset for the PyTorch version of the model obtained after running iterationtwo.ipynb. It also performs inference on the same dataset using the optimized and quantized ONNX versions of the same model, and compares the performance of the models evaluated on the same test dataset using ONNX profiling tools.
inclues functionality for user to enter file path to test dataset, model parameters, and new locations to which the quantized and ONNX optimized models can be written to and stored.

inference.py: It loads the fine-tuned PyTorch model, processes a test dataset, and evaluates its performance. It also converts the PyTorch model to the ONNX format, then optimizes it for inference using ONNX Runtime. It then compares the performance (accuracy, precision, recall, F1 score) of both the PyTorch and optimized ONNX models.

requirements.txt: lists the requirements needed to build Docker image from inference.py, using Dockerfile

Dockerfile: Docker file for building Docker image from inference.py

fine_tuned_efficientnet.zip: Stores the weights and biases of the fine-tuned model's parameters-the model which was used for Pytorch inference
