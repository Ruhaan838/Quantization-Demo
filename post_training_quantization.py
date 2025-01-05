import torch
import torch.ao.quantization
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

from tqdm import tqdm
from pathlib import Path
import os

torch.manual_seed(42)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = "cpu"
print(f"Using device: {device}")

class SimpleNN(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=200):
        super().__init__()
        self.layer_1 = nn.Linear(28*28, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x


def train(model, dataloader, epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        losses = []
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for data in tqdm(dataloader, desc=f"Training", ncols=50):
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
    return losses

def model_size(model: nn.Module):
    torch.save(model.state_dict(), "temp_model.p")
    size_kb = os.path.getsize("temp_model.p")/1e3
    os.remove("temp_model.p")
    
    print(f"\nModel Size: {size_kb:.2f} KB")
    return size_kb

@torch.no_grad()
def test(model, dataloader):
    correct = 0
    total = 0
    predictions = []
    
    model.eval()
    for data in tqdm(dataloader, desc='Testing', ncols=50):
        x, y = data
        x, y = x.to(device), y.to(device)
        output = model(x)
        predictions.extend(torch.argmax(output, dim=1).cpu().numpy())
        correct += (torch.argmax(output, dim=1) == y).sum().item()
        total += y.size(0)
    
    acc = correct / total
    print(f"\nTest Accuracy: {acc:.2%}")
    return acc

model = SimpleNN().to(device)
model_name = 'simpleNN.pt'

if Path(model_name).exists():
    model.load_state_dict(torch.load(model_name, weights_only=False))
    print("\nLoading saved model...")
else:
    print("\nTraining new model...")
    train(model, train_dataloader, epochs=3)
    torch.save(model.state_dict(), model_name)

print("\n" + "="*50)
print("Original Model Evaluation")
print("="*50)
original_size = model_size(model)
original_acc = test(model, test_dataloader)

class QuantizedSimpleNN(nn.Module):
    def __init__(self, hidden_size_1=100, hidden_size_2=200):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.layer_1 = nn.Linear(28*28, hidden_size_1)
        self.layer_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer_3 = nn.Linear(hidden_size_2, 10)
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.quant(x)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.layer_3(x)
        x = self.dequant(x)
        return x

print("\n" + "="*50)
print("Quantized Model Evaluation")
print("="*50)

model_Q = QuantizedSimpleNN().to(device)
model_Q.load_state_dict(model.state_dict())
model_Q.eval()

torch.backends.quantized.engine = 'qnnpack'
model_Q.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
model_Q = torch.ao.quantization.prepare(model_Q)
print("\nTesting Prepared Quantized Model...")
quant_prep_acc = test(model_Q, test_dataloader)

model_Q = torch.ao.quantization.convert(model_Q)
print("\nTesting Final Quantized Model...")
quant_size = model_size(model_Q)
quant_acc = test(model_Q, test_dataloader)
