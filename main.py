import os
import cv2
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import json
import pickle
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from swin_transformer import SwinTransformer


cv2.setNumThreads(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the label data from given directory. Typically the label data for out dataset is called metadata.json.
# The output of this function is a list with metadata
# [Filename, Label]
def LabelLoader(data_path=None):
    metadata_path = os.path.join(data_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Can't find metadata: {metadata_path}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    labels = {vid: 1 if metadata[vid]["label"] == "FAKE" else 0 for vid in metadata}
    return metadata, list(metadata.keys()), labels


class VideoDataset(Dataset):
    def __init__(self, metadata, video_dir=None, frames_per_video=16, transform=None):
        self.metadata = metadata
        self.video_dir = video_dir
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.video_files = list(metadata.keys())
        self.labels = {vid: 1 if metadata[vid]["label"] == "FAKE" else 0 for vid in self.video_files}

    def extract_frames(self, video_path):
        # print("Opening file:", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Can't open file:", video_path)
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count < self.frames_per_video:
            # Repeat the last frame if frame not enough (should not happen in training sample for 18 frames)
            sample_idxs = np.linspace(0, frame_count - 1, frame_count, dtype=int).tolist()
            while len(sample_idxs) < self.frames_per_video:
                sample_idxs.append(sample_idxs[-1])
        else:
            sample_idxs = np.linspace(0, frame_count - 1, self.frames_per_video, dtype=int)

        frames = []
        for idx in sample_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                print("Sampling Failed =", idx, "Video:", video_path)
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) < self.frames_per_video:
            print("no frame", os.path.basename(video_path))
            return None
        return np.array(frames)

    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_name)
        label = torch.tensor(self.labels[video_name], dtype=torch.float32)

        frames = self.extract_frames(video_path)
        if frames is None:
            print("Not enough frames:", video_name)
            frames = np.zeros((self.frames_per_video, 224, 224, 3), dtype=np.uint8)

        if self.transform:
            frames = np.stack([self.transform(frame) for frame in frames])
        
        frames = torch.tensor(frames, dtype=torch.float32).permute(1, 0, 2, 3)
        return frames, label

    def __len__(self):
        return len(self.video_files)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Third conv layer
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels * self.expansion)

        # ReLU layer to activate non-linear features, LeakyReLU to avoid too small gradient
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        # Downsample if needed (mostly first block of each layer)
        self.downsample = downsample

        self.stride = stride

    def forward(self, x):
        identity = x    # enable residual connection between each blocks
        # Adjust dimension if needed
        if self.downsample is not None:     
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += identity   # add residual connection
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # 7x7, 64, /2       224x224x3 to 112x112x64
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.LeakyReLU(0.1)

        # pool, /2      112x112x64 to 56x56x64
        self.pool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        # First Residual Block (64 channels x3)    56x56x64 to 56x56x64
        # We don't need downsample for the first block because the input basically is downsample.
        self.layer1 = self.make_layer(64, 64, num_blocks=3, stride=1)

        # Second Residual Block (128 channels x4)      56x56x64 to 28x28x128, so now we need downsample
        self.layer2 = self.make_layer(64*Bottleneck.expansion, 128, num_blocks=4, stride=2)

        # # Third Residual Block (256 channels x6)        28x28x128 to 14x14x256,  downsample
        # self.layer3 = self.make_layer(128*Bottleneck.expansion, 256, num_blocks=6, stride=2)

        # # Fourth Residual Block (512 channels x3)       14x14x256 to 7x7x512,   downsample
        # self.layer4 = self.make_layer(256*Bottleneck.expansion, 512, num_blocks=3, stride=2)

        # # Here I deleted the pooling and fc for the transformer
        # # Global Average Pooling
        # self.global_avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        # self.global_time_pool = nn.AdaptiveAvgPool1d(1) 

        # # Fully Connected Layer
        # self.fc = nn.Linear(512*Bottleneck.expansion, 1)

        # d_model = 512 * Bottleneck.expansion
        # self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        # self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)

        # self.fc = nn.Linear(d_model, 1)

    def make_layer(self, in_channels, out_channels, num_blocks, stride):
        downsample = None   # Reshape the identity to the corrent dimension
        if stride != 1 or in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride, downsample))

        for _ in range(1, num_blocks):
            layers.append(Bottleneck(out_channels * Bottleneck.expansion, out_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        # # Deleted pool and fc for transformer
        # x = self.global_avg_pool(x)  # [batch, 2048, T, 1, 1]
        # x = x.squeeze(-1).squeeze(-1)  # [batch, 2048, T]
        # x = self.global_time_pool(x)  # [batch, 2048, 1]
        # x = x.squeeze(-1)  # [batch, 2048]
        # x = self.fc(x)

        # B, C, T, H, W = x.shape
        # x = nn.AdaptiveAvgPool3d((T, 1, 1))(x)      # [B, C, T, 1, 1]
        # x = x.view(B, C, T)       # [B, C, T]
        # x = x.permute(0, 2, 1)     # [B, T, C]

        # x = self.local_attn(x)

        # x = self.transformer_encoder(x)  # [B, T, C]

        # x = x.mean(dim=1)  # [B, C]

        # x = self.fc(x)    # [B, 1]
        return x

def compute_mean_std(dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for frames, _ in dataloader:
        frames = frames.view(frames.shape[0], frames.shape[1], -1)  # (B, C, T, H, W) -> (B, C, T*H*W)
        mean += frames.mean(dim=[0, 2])
        std += frames.std(dim=[0, 2])
        total_samples += 1

    mean /= total_samples
    std /= total_samples
    return mean.numpy(), std.numpy()


def evaluate_model(model, dataloader, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.to(device)
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)

            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).long()
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
    print(len(all_preds), all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        roc_auc = None

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    print("Model Evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    print("\nCN")
    print(cm)
    print("\nReport")
    print(report)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }
    
    return metrics

# Cache some test sample to local device for testing. Save some time
def cache_testset_separately(metadata, video_dir, save_dir, frames_per_video=16, transform=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    dataset = VideoDataset(metadata, video_dir, frames_per_video, transform)
    
    for idx in range(len(dataset)):
        frames, label = dataset[idx]
        video_name = dataset.video_files[idx]
        sample_file = os.path.join(save_dir, f"{video_name}.pkl")
        with open(sample_file, "wb") as f:
            pickle.dump((frames, label), f)
        if idx % 10 == 0:
            print(f"Loaded {idx} videos")
    print("Cache Completed!")

class CachedTestDataset(Dataset):
    def __init__(self, cached_dir):
        self.cached_files = [
            os.path.join(cached_dir, fname)
            for fname in os.listdir(cached_dir)
            if fname.endswith('.pkl')
        ]
        
    def __getitem__(self, idx):
        sample_file = self.cached_files[idx]
        with open(sample_file, "rb") as f:
            frames, label = pickle.load(f)
        return frames, label

    def __len__(self):
        return len(self.cached_files)

def load_cached_testset(save_file):
    with open(save_file, "rb") as f:
        cached_data = pickle.load(f)
    print(f"From {save_file} loaded {len(cached_data)} samples")
    return cached_data


class ResNet_Swin(nn.Module):
    def __init__(self, embed_dim=96, num_classes=1):
        super(ResNet_Swin, self).__init__()

        self.resnet = ResNet()  # (B, 2048, T, 7, 7)
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        
        self.swin_transformer = SwinTransformer(
            img_size=28,
            patch_size=1,
            in_chans=embed_dim,
            num_classes=0,
            embed_dim=embed_dim,
            depths=[2],
            num_heads=[3],
            window_size=7,
            mlp_ratio=4.0,
            dropout=0.0
        )
        
        self.temporal_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.temporal_transformer = nn.TransformerEncoder(self.temporal_layer, num_layers=2)
        
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):

        x = self.resnet(x)
        B, C, T, H, W = x.shape
        
        # each frame as a image, process into swin
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # (B*T, 2048, 28, 28)
        
        x = self.proj(x)  # (B*T, embed_dim, 7, 7)
        
        x = self.swin_transformer.patch_embed(x)   # (B*T, num_patches, embed_dim)  num_patches = 7*7 = 49
        x = self.swin_transformer.pos_drop(x)
        for layer in self.swin_transformer.layers:
            x = layer(x)
        x = self.swin_transformer.norm(x)

        x = x.mean(dim=1)
        
        x = x.view(B, T, -1)
        
        x = self.temporal_transformer(x)  # (B, T, embed_dim)
        
        x = x.mean(dim=1)  # (B, embed_dim)
        
        x = self.fc(x)  # (B, num_classes)
        return x



def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_video_dir = os.path.join(base_dir, "Balanced Sample")
    train_metadata, train_video_files, train_labels = LabelLoader(data_path=train_video_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.46677276, 0.4492136,  0.38623506], std=[0.2535491,  0.25852382, 0.24249125])
    ])

    train_dataset = VideoDataset(train_metadata, train_video_dir, transform=transform)
    dataset_size = len(train_dataset)

    # sample_batch = next(iter(train_loader))
    # videos, labels = sample_batch
    # print("Videos shape:", videos.shape)
    # print("Labels shape:", labels.shape)

    # frame = videos[0, :, 0].permute(1, 2, 0)
    # frame = frame.numpy()
    # plt.imshow(frame)
    # plt.show()

    model = ResNet_Swin()
    # model.load_state_dict(torch.load("model_weights3.pth", map_location=device))
    model.to(device)

    # pos_weight = torch.tensor([1/1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 20
    batch_size = 16
    num_workers = 4
    for epoch in range(num_epochs):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        split = int(0.9 * dataset_size)
        train_indices, val_indices = indices[:split], indices[split:]

        train_subset = Subset(train_dataset, train_indices)
        val_subset   = Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        
        model.train()
        running_loss = 0.0
        begin = True
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device).view(-1, 1).float()

            optimizer.zero_grad()
            outputs = model(videos)  # [batch_size, 1]
            loss = criterion(outputs, labels)

            loss.backward()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
            optimizer.step()

            running_loss += loss.item()
            if begin and epoch==0:
                print("Correct!")
                begin=False
            
        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for videos, labels in val_loader:
                videos, labels = videos.to(device), labels.to(device).view(-1, 1).float()
                outputs = model(videos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    SAVE_PATH = "model_weights.pth"
    torch.save(model.state_dict(), SAVE_PATH)
    print("Model saved!")

def test():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_video_dir = os.path.join(base_dir, "dfdc_train_part_0")
    test_metadata, test_video_files, test_labels = LabelLoader(data_path=test_video_dir)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.46677276, 0.4492136,  0.38623506], std=[0.2535491,  0.25852382, 0.24249125])
    ])

    cache_dir = os.path.join(base_dir, "cached_test_samples")
    
    if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
        cache_testset_separately(test_metadata, test_video_dir, cache_dir, frames_per_video=16, transform=transform)
    
    cached_dataset = CachedTestDataset(cache_dir)
    test_loader = torch.utils.data.DataLoader(cached_dataset, batch_size=16, shuffle=False, num_workers=4)

    model = ResNet_Swin()
    model.load_state_dict(torch.load("model_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    metrics = evaluate_model(model, test_loader)
    
    print("Final Result:", metrics)

if __name__ == '__main__':
    train()
    test()
