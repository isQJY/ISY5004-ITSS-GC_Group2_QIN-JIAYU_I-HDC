import argparse
import torch
from models.i_hdc import i_hdc
from utils.data_loader import get_data_loaders
from utils.metrics import compute_accuracy, compute_confusion_rate

def evaluate(model_path, dataset, data_dir, image_size=64, num_classes=43, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    _, test_loader = get_data_loaders(dataset, data_dir, batch_size=64, image_size=image_size)
    
    model = i_hdc(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_acc, total_confusion = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_acc += compute_accuracy(outputs, labels) * inputs.size(0)
            total_confusion += compute_confusion_rate(outputs, labels) * inputs.size(0)
    
    total_acc /= len(test_loader.dataset)
    total_confusion /= len(test_loader.dataset)
    print(f'Test Accuracy: {total_acc:.4f}, Test Confusion Rate: {total_confusion:.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate i-HDC model.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['gtsrb', 'cifar10'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=43)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    evaluate(args.model_path, args.dataset, args.data_dir, args.image_size, args.num_classes, args.device)
