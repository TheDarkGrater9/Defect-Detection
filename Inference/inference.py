#!/usr/bin/env python3
import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import psutil
import onnx
import onnxruntime as ort
import argparse
from torchvision.models import efficientnet_b0

class DefectDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [f for f in os.listdir(folder) if f.endswith('.png') and not f.endswith('_GT.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder, self.images[idx])
        gt_name = os.path.join(self.folder, self.images[idx].replace('.png', '_GT.png'))
        image = Image.open(img_name).convert('RGB')
        label = 0
        if os.path.exists(gt_name):
            label_image = plt.imread(gt_name)
            label = int(np.max(label_image) > 0)
        if self.transform:
            image = self.transform(image)
        return image, label

def calculate_metrics(true_labels, pred_labels):
    true_labels = np.array(true_labels).astype(int)
    pred_labels = np.array(pred_labels).astype(int)
    if len(np.unique(true_labels)) > 2 or len(np.unique(pred_labels)) > 2:
        raise ValueError("Labels must be binary (0 or 1). Found more than two classes.")
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=1)
    recall = recall_score(true_labels, pred_labels, zero_division=1)
    f1 = f1_score(true_labels, pred_labels, zero_division=1)
    cm = confusion_matrix(true_labels, pred_labels)
    return accuracy, precision, recall, f1, cm

def pytorch_inference(model_path, test_folder, device):
    image_size = (224, 224)
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_dataset = DefectDataset(test_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    true_labels, pred_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs).cpu().numpy().round().astype(int)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.flatten())
    end_time = time.time()

    metrics = calculate_metrics(true_labels, pred_labels)
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    return metrics, end_time - start_time, memory_usage

def export_to_onnx(pytorch_model_path, onnx_model_path, device):
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    state_dict = torch.load(pytorch_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    if not onnx_model_path.endswith(".onnx"):
        onnx_model_path += ".onnx"

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX model exported to {onnx_model_path}")
    return onnx_model_path

def optimized_onnx_inference(onnx_model_path, test_folder):
    image_size = (224, 224)
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_dataset = DefectDataset(test_folder, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_model_path, session_options, providers=["CPUExecutionProvider"])

    true_labels, pred_labels = [], []
    start_time = time.time()
    for inputs, labels in test_loader:
        inputs = inputs.numpy()
        ort_inputs = {ort_session.get_inputs()[0].name: inputs}
        ort_outs = ort_session.run(None, ort_inputs)
        sigmoid_outputs = 1 / (1 + np.exp(-ort_outs[0]))
        preds = np.round(sigmoid_outputs).astype(int)
        true_labels.extend(labels.numpy())
        pred_labels.extend(preds.flatten())
    end_time = time.time()

    metrics = calculate_metrics(true_labels, pred_labels)
    memory_usage = psutil.virtual_memory().used / (1024 ** 3)
    return metrics, end_time - start_time, memory_usage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect Detection Inference Script")
    parser.add_argument('--model', required=True, help="Path to the PyTorch .pth model file")
    parser.add_argument('--test', required=True, help="Path to the test image folder")
    parser.add_argument('--export_onnx', default="/app/model.onnx", help="Path to save exported ONNX model")
    args = parser.parse_args()

    device = torch.device("cpu")

    pytorch_metrics, pytorch_time, pytorch_memory = pytorch_inference(args.model, args.test, device)
    onnx_model_path = export_to_onnx(args.model, args.export_onnx, device)
    optimized_metrics, optimized_time, optimized_memory = optimized_onnx_inference(onnx_model_path, args.test)

    print("\nPyTorch Inference Results:")
    print(f"Accuracy: {pytorch_metrics[0]:.4f}, Precision: {pytorch_metrics[1]:.4f}, Recall: {pytorch_metrics[2]:.4f}, F1 Score: {pytorch_metrics[3]:.4f}")
    print(f"Time Taken: {pytorch_time:.2f}s, Memory Usage: {pytorch_memory:.2f}GB")

    print("\nOptimized ONNX Inference Results:")
    print(f"Accuracy: {optimized_metrics[0]:.4f}, Precision: {optimized_metrics[1]:.4f}, Recall: {optimized_metrics[2]:.4f}, F1 Score: {optimized_metrics[3]:.4f}")
    print(f"Time Taken: {optimized_time:.2f}s, Memory Usage: {optimized_memory:.2f}GB")
