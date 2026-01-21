# derin-renme0
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- GLOBAL TANIMLAMALAR (Sınıflar ve Fonksiyonlar Burada Kalmalı) ---

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet34_Original(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34_Original, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64,  3)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class TransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self): return len(self.subset)

def unnormalize(tensor):
    tensor = tensor.cpu().clone()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

# --- ANA ÇALIŞTIRMA BLOĞU ---
# Windows'ta multiprocessing hatasını önlemek için kodun geri kalanı burada olmalı.
if __name__ == '__main__':

    # --- 1. AYARLAR ---
    USE_TRANSFER_LEARNING = True                                         #TRANSFER  LEARNİNG RESNET34
    DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/apple"
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- SİSTEM AYARLARI ---")
    print(f"Cihaz: {device}")
    print(f"Mod: {'Transfer Learning' if USE_TRANSFER_LEARNING else 'Sıfırdan Eğitim (Custom ResNet34)'}")
    print("-" * 30)

    # --- 2. VERİ HAZIRLAMA ---
    try:
        raw_dataset = datasets.ImageFolder(DATA_DIR)
        classes = raw_dataset.classes
        num_classes = len(classes)
    except Exception as e:
        print(f"HATA: Veri seti okunamadı. {e}")
        exit()

    # Veri Bölme
    total_len = len(raw_dataset)
    train_len = int(total_len * 0.70)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len

    train_subset, val_subset, test_subset = random_split(
        raw_dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    # Tablo Oluşturma
    def create_table(full_ds, tr_sub, val_sub, te_sub, class_names):
        stats = {cls: {'Train': 0, 'Val': 0, 'Test': 0} for cls in class_names}
        def fill(subset, col):
            for idx in subset.indices:
                label = full_ds.targets[idx]
                stats[class_names[label]][col] += 1
        fill(tr_sub, 'Train')
        fill(val_sub, 'Val')
        fill(te_sub, 'Test')

        df = pd.DataFrame.from_dict(stats, orient='index')
        df['Total'] = df.sum(axis=1)
        df = df[['Total', 'Train', 'Val', 'Test']]
        df.loc['GENEL TOPLAM'] = df.sum()
        return df

    print("\n--- Veri Seti Dağılım Tablosu ---")
    print(create_table(raw_dataset, train_subset, val_subset, test_subset, classes))
    print("-" * 50)

    # Augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = TransformSubset(train_subset, transform=train_transforms)
    val_data = TransformSubset(val_subset, transform=val_test_transforms)
    test_data = TransformSubset(test_subset, transform=val_test_transforms)

    # num_workers=2 Windows'ta sadece if __name__ == '__main__': içindeyse çalışır
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 3. MODEL MİMARİSİ ---
    if USE_TRANSFER_LEARNING:
        print("\n--- Model: Pretrained ResNet-34 ---")
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print("\n--- Model: Custom ResNet-34 (Original Structure) ---")
        model = ResNet34_Original(num_classes)

    model = model.to(device)

    print("\n--- DETAYLI MODEL MİMARİSİ ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nToplam Parametre Sayısı: {total_params:,}")
    print("-" * 40)

    # --- 4. OPTIMIZER VE EĞİTİM ---
    criterion = nn.CrossEntropyLoss()

    if USE_TRANSFER_LEARNING:
        optimizer = optim.Adam(model.fc.parameters(), lr=0.0001, weight_decay=WEIGHT_DECAY)
        scheduler = None
    else:
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Eğitim Başlıyor ({NUM_EPOCHS} Epoch) ---")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        ep_loss = run_loss / len(train_loader.dataset)
        ep_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, pred = torch.max(outputs, 1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

        ep_val_loss = val_loss / len(val_loader.dataset)
        ep_val_acc = 100 * val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(ep_val_loss)

        history['train_loss'].append(ep_loss)
        history['train_acc'].append(ep_acc)
        history['val_loss'].append(ep_val_loss)
        history['val_acc'].append(ep_val_acc)

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | LR: {current_lr:.6f} | "
              f"Train Loss: {ep_loss:.4f} Acc: {ep_acc:.2f}% | "
              f"Val Loss: {ep_val_loss:.4f} Acc: {ep_val_acc:.2f}%")

    print(f"\nEğitim Tamamlandı. Süre: {(time.time() - start_time)/60:.2f} dk")

    # --- 5. GRAFİKLER VE SONUÇLAR ---
    plt.figure(figsize=(14, 5))
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-o', label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-o', label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], 'r-o', label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Test Raporu ---")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.show()

    # Örnek Gösterim
    try:
        inputs, labels = next(iter(test_loader))
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        plt.figure(figsize=(12, 4))
        for i in range(min(4, len(inputs))):
            ax = plt.subplot(1, 4, i + 1)
            img = unnormalize(inputs[i]).permute(1, 2, 0).numpy()
            plt.imshow(img)
            true_lbl = classes[labels[i].item()]
            pred_lbl = classes[preds[i].item()]
            col = 'green' if true_lbl == pred_lbl else 'red'
            plt.title(f"G: {true_lbl}\nT: {pred_lbl}", color=col)
            plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Görselleştirme hatası: {e}")
        
