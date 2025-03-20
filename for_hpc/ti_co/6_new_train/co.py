
#region Import
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset 
from timm.data.mixup import Mixup
from timm.data.auto_augment import rand_augment_transform
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchvision.transforms.autoaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import math
import numpy as np
from tqdm import tqdm
import os
import copy
import glob 
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image as ToPILImage
from PIL import Image, ImageFile
from einops import rearrange
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings("ignore")
#endregion

#region Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

num_classes = 200
num_workers = 2
#endregion

#region 1. KONFIGURASI HYPERPARAMETER
CONFIG = {
    # Hyperparameter Preprocessing dan Augmentasi Data
    "center_crop": True,                     # Crop gambar dari tengah
    "randaugment_n": 2,                      # Jumlah operasi RandAugment
    "randaugment_m": 15,                     # Magnitude RandAugment
    "mixup_alpha": 0.8,                      # Parameter alpha untuk Mixup
    
    # Hyperparameter Fungsi Loss
    "loss_type": "Softmax",                  # Jenis fungsi loss
    "label_smoothing": 0.1,                  # Smoothing factor untuk label
    
    # Hyperparameter Training
    "train_epochs": 300,                     # Jumlah epoch
    "train_batch_size": 20,                # Ukuran batch
    
    # Hyperparameter Optimisasi
    "optimizer_type": "AdamW",               # Jenis optimizer
    "peak_lr": 1e-3,                         # Learning rate maksimum
    "min_lr": 1e-5,                          # Learning rate minimum
    "warmup_steps": 10000,                   # Jumlah langkah warm-up
    "lr_decay_schedule": "Cosine",           # Jadwal penurunan learning rate
    "weight_decay_rate": 0.05,               # Rate weight decay
    "gradient_clip": 1.0,                    # Nilai maksimum gradien
    "ema_decay_rate": None                   # EMA dinonaktifkan
}
#endregion

#region 2. DEFINISI ARSITEKTUR MODEL (CoAtNet)

#region CoatNet
def conv_3x3_bn(inp, oup, image_size, downsample=False):
    stride = 1 if downsample == False else 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.GELU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm):
        super().__init__()
        self.norm = norm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MBConv(nn.Module):
    def __init__(self, inp, oup, image_size, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        if self.downsample:
            self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, stride, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1,
                          groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv = PreNorm(inp, self.conv, nn.BatchNorm2d)

    def forward(self, x):
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(x)
        else:
            return x + self.conv(x)


class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(inp, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, oup),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih*self.iw, w=self.ih*self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, downsample=False, dropout=0.):
        super().__init__()
        hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2d(3, 2, 1)
            self.pool2 = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.ff = FeedForward(oup, hidden_dim, dropout)

        self.attn = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(inp, self.attn, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

        self.ff = nn.Sequential(
            Rearrange('b c ih iw -> b (ih iw) c'),
            PreNorm(oup, self.ff, nn.LayerNorm),
            Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        )

    def forward(self, x):
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return x


class CoAtNet(nn.Module):
    def __init__(self, image_size, in_channels, num_blocks, channels, num_classes=num_classes, block_types=['C', 'C', 'T', 'T']):
        super().__init__()
        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2))
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4))
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8))
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16))
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32))

        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x).view(-1, x.shape[1])
        x = self.fc(x)
        return x

    def _make_layer(self, block, inp, oup, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(inp, oup, image_size, downsample=True))
            else:
                layers.append(block(oup, oup, image_size))
        return nn.Sequential(*layers)


def coatnet_0():
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_1():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_2():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_3():
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_4():
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(1, 3, 224, 224)

    net = coatnet_0()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_1()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_2()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_3()
    out = net(img)
    print(out.shape, count_parameters(net))

    net = coatnet_4()
    out = net(img)
    print(out.shape, count_parameters(net))
#endregion

#endregion

#region 3. KONFIGURASI DATA PIPELINE
train_transform = transforms.Compose([
    RandAugment(num_ops=CONFIG['randaugment_n'], magnitude=CONFIG['randaugment_m']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TrainTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/train/*/*/*.JPEG")
        self.transform = transform

        self.class_to_idx = {}
        class_folders = set([path.split('/')[5] for path in self.filenames])
        for i, class_name in enumerate(sorted(class_folders)):
            self.class_to_idx[class_name] = i

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)

        class_name = img_path.split('/')[5]
        label = self.class_to_idx[class_name]

        image = ToPILImage(image)
        if self.transform:
            image = self.transform(image)
        return image, label

class ValTinyImageNetDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/val/images/*.JPEG")
        self.transform = transform

        train_files = glob.glob("/home/tasi2425111/resized-tiny-imagenet-200/train/*/*/*.JPEG")
        class_folders = set([path.split('/')[5] for path in train_files])
        self.class_to_idx = {}
        for i, class_name in enumerate(sorted(class_folders)):
            self.class_to_idx[class_name] = i

        self.cls_dic = {}
        for i, line in enumerate(open('/home/tasi2425111/resized-tiny-imagenet-200/val/val_annotations.txt', 'r')):
            a = line.split('\t')
            img, cls_id = a[0], a[1]
            self.cls_dic[img] = self.class_to_idx[cls_id]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = read_image(img_path)
        if image.shape[0] == 1:
            image = read_image(img_path, ImageReadMode.RGB)

        label = self.cls_dic[img_path.split('/')[-1]]
        image = ToPILImage(image)
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = TrainTinyImageNetDataset(transform = train_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["train_batch_size"], shuffle=True, num_workers=num_workers, pin_memory=True)

val_dataset = ValTinyImageNetDataset(transform = val_transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=CONFIG["train_batch_size"], shuffle=False, num_workers=num_workers, pin_memory=True)

#endregion

#region 4. KONFIGURASI MIXUP
def get_mixup_fn():
    mixup_fn = Mixup(
        mixup_alpha=CONFIG["mixup_alpha"],
        label_smoothing=CONFIG["label_smoothing"],
        num_classes=num_classes  # Sesuaikan dengan dataset Anda
    )
    return mixup_fn
#endregion

#region 5. KONFIGURASI LOSS FUNCTION
def get_loss_fn():
    if CONFIG["loss_type"] == "Softmax":
        # CrossEntropyLoss sudah mencakup softmax
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    else:
        # Tambahkan loss function lain jika diperlukan
        criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    
    return criterion
#endregion

#region 6. KONFIGURASI OPTIMIZER
def get_optimizer(model):
    if CONFIG["optimizer_type"] == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG["peak_lr"],
            weight_decay=CONFIG["weight_decay_rate"]
        )
    else:
        # Fallback ke Adam jika AdamW tidak tersedia
        optimizer = optim.Adam(
            model.parameters(),
            lr=CONFIG["peak_lr"],
            weight_decay=CONFIG["weight_decay_rate"]
        )
    
    return optimizer
#endregion

#region 7. KONFIGURASI LEARNING RATE SCHEDULER
def get_lr_scheduler(optimizer, num_training_steps):
    if CONFIG["lr_decay_schedule"] == "Cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_training_steps,  # Total langkah training
            lr_min=CONFIG["min_lr"],
            warmup_t=CONFIG["warmup_steps"],
            warmup_lr_init=CONFIG["min_lr"],
            cycle_limit=1
        )
    else:
        # Fallback ke StepLR
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return lr_scheduler
#endregion

#region 8. FUNGSI HELPER UNTUK EMA (EXPONENTIAL MOVING AVERAGE)
class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')
        
        # Fix EMA. https://github.com/pytorch/vision/pull/2591
        for param in self.ema.parameters():
            param.requires_grad_(False)
    
    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            
            for k, v in esd.items():
                if needs_module:
                    k = 'module.' + k
                
                model_v = msd[k].detach()
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + (1. - self.decay) * model_v)
                else:
                    v.copy_(model_v)
#endregion

#region 9. FUNGSI TRAINING
def train_model():    
    # Get model using coatnet_3 function
    model = coatnet_3() 
    model = model.to(device)
    
    mixup_fn = get_mixup_fn()
    criterion = get_loss_fn()
    optimizer = get_optimizer(model)
    
    # Calculate total number of training steps
    total_steps = len(train_loader) * CONFIG["train_epochs"]
    lr_scheduler = get_lr_scheduler(optimizer, total_steps)
    
    # Setup EMA if enabled
    ema_model = None
    if CONFIG["ema_decay_rate"] is not None:
        ema_model = ModelEMA(model, decay=CONFIG["ema_decay_rate"])
    
    # Setup gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Early stopping variables
    patience = 5
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # Lists to store metrics for summary
    val_losses = []
    val_accs = []
    
    # Training loop
    for epoch in range(CONFIG["train_epochs"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['train_epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply mixup if enabled
            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if CONFIG["gradient_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model if enabled
            if ema_model is not None:
                ema_model.update(model)
            
            # Update learning rate
            lr_scheduler.step_update(epoch * len(train_loader) + batch_idx)
            
            # Update statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if isinstance(targets, torch.Tensor) and targets.dim() == 1:
                correct += predicted.eq(targets).sum().item()
            
            progress_bar.set_postfix({
                'loss': train_loss/(batch_idx+1), 
                'lr': optimizer.param_groups[0]['lr']
            })
        
        # Validate after each epoch
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Use EMA model for validation if available
                if ema_model is not None:
                    outputs = ema_model.ema(inputs)
                else:
                    outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        # Check if this is the best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            
            # Save best model state
            if ema_model is not None:
                best_model_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                }
            else:
                best_model_state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
                    'val_loss': epoch_val_loss,
                    'val_acc': epoch_val_acc,
                }
                
            # Save the best model checkpoint
            torch.save(best_model_state, 'best_model.pth')
            print(f"Saved new best model with validation loss: {epoch_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{patience}")
            
        # Check if early stopping criteria is met
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler.state_dict() if hasattr(lr_scheduler, 'state_dict') else None,
        'ema': ema_model.ema.state_dict() if ema_model is not None else None,
        'val_acc': epoch_val_acc,
    }
    
    torch.save(checkpoint, 'final_model.pth')
    
    # Print training summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Total epochs: {epoch+1}")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_model_state['epoch']})")
    print(f"Final metrics:")
    print(f"  Validation Loss: {val_losses[-1]:.4f}")
    print(f"  Validation Accuracy: {val_accs[-1]:.2f}%")
    print("="*50)
#endregion

train_model()