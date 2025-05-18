import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from models.i_hdc import i_hdc
from utils.data_loader import get_data_loaders
from utils.metrics import compute_accuracy, compute_confusion_rate

def train(config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders(
        config['dataset']['name'],
        config['dataset']['data_dir'],
        config['training']['batch_size'],
        config['dataset']['image_size']
    )
    
    model = i_hdc(
        num_classes=config['dataset']['num_classes'],
        base_channels=config['model']['num_channels'],
        dropout_rate=config['model']['dropout_rate'],
        stochastic_depth_prob=config['model']['stochastic_depth_prob']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                           weight_decay=config['training']['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=config['training']['scheduler']['factor'],
                                 patience=config['training']['scheduler']['patience'])
    
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["training"]["epochs"]}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_acc += compute_accuracy(outputs, labels) * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_acc /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_acc, val_confusion = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_acc += compute_accuracy(outputs, labels) * inputs.size(0)
                val_confusion += compute_confusion_rate(outputs, labels) * inputs.size(0)
        
        val_acc /= len(test_loader.dataset)
        val_confusion /= len(test_loader.dataset)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Val Confusion: {val_confusion:.4f}')
        
        scheduler.step(val_acc)
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint']['save_freq'] == 0:
            checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f'i_hdc_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train i-HDC model.')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)
