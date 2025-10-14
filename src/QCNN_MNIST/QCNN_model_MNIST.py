import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pennylane as qml
from torch.utils.data import Subset
from functorch import vmap

n_layers = 1

class QCNN(nn.Module):
    
    def __init__(self,image,rand_params):
        super(QCNN, self).__init__()
        self.image = image
        self.rand_params = nn.Parameter(torch.tensor(rand_params, dtype=torch.float32))
        
        dev = qml.device("default.qubit", wires=4)
        @qml.qnode(dev, interface='torch')
        def circuit(image,rand_params):
            # Random quantum circuit (initial layer)
            pixels = []
            for _ in range(15): #determine how many times you want to rerun - gives better results
                for pixel in image:
                    pixels.append(pixel)
                    if len(pixels) % 3 == 0 and len(pixels) != 0:
                        for j in range(4): #using 4 qubits
                            qml.Rot(phi=pixels[0]*np.pi,theta=pixels[1]*np.pi,omega = pixels[2]*np.pi, wires=j) # Apply rotation based on the input value
                            # Entangling layer
                            for k in range(3):
                                qml.CNOT(wires=[k, k+1])
                            qml.CNOT(wires=[3,0])
                        pixels = []
            qml.RandomLayers(rand_params, wires=list(range(4)))

            # Measurement producing 4 classical output values
            return [qml.expval(qml.PauliZ(wires=j)) for j in range(4)]

        self.circuit = circuit
        self.fc = nn.Linear(14 * 14 * 4, 10)
    
    def forward(self, images):
        batch_size, _, height_image, width_image = images.shape

        all_blocks = []
        for b in range(batch_size):
            blocks = []
            image = images[b]
            for j in range(0, height_image-1, 2):
                for k in range(0, width_image-1, 2):
                    blocks.append(torch.stack([
                        image[0, j, k],
                        image[0, j, k+1],
                        image[0, j+1, k],
                        image[0, j+1, k+1]
                    ]))
            blocks = torch.stack(blocks)
            all_blocks.append(blocks)
        
        all_blocks = torch.stack(all_blocks)
        
        block_vmap = vmap(self.circuit, in_dims=(0, None))  
        batch_vmap = vmap(block_vmap, in_dims=(0, None))
        q_results = batch_vmap(all_blocks, self.rand_params)
        
        if isinstance(q_results, list):
            q_results = torch.stack([torch.tensor(x) for x in q_results])
            
        q_results = q_results.to(torch.float32)
        q_results = q_results.reshape(batch_size, -1)
        outputs = self.fc(q_results)
        
        return F.log_softmax(outputs, dim=1)
    

def train(args, model, device, dataset, optimizer, epoch):
    model.train()
    for i in range(len(dataset)):
        image, label = dataset[i]
        image = image.unsqueeze(0).to(device)         # shape: (1, 1, 28, 28)
        label = torch.tensor([label], device=device)  # shape: (1,)

        optimizer.zero_grad()
        output = model(image)                          # forward pass
        loss = F.nll_loss(output, label)              # compute loss
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, i+1 , len(dataset),
                100*(i+1) / len(dataset), loss.item()
                ))


def test(model, device, dataset):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            image, label = dataset[i]  # test sample
            image = image.unsqueeze(0).to(device)  # shape (1, 1, 28, 28)
            label = torch.tensor([label], device=device)
            
            output = model(image)
            test_loss += F.nll_loss(output, label, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            #print("Predicted class:", pred.item(), "True label:", label.item())
            
        test_loss /= len(dataset)
        accuracy = 100. * correct / len(dataset)
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(dataset),
            accuracy))
    return test_loss, accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_accel = not args.no_accel and torch.accelerator.is_available()

    torch.manual_seed(args.seed)

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_accel:
        accel_kwargs = {'num_workers': 1,
                        'persistent_workers': True,
                       'pin_memory': True,
                       'shuffle': True}
        
        train_kwargs.update(accel_kwargs)
        test_kwargs.update(accel_kwargs)

    transform=transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    train_subset = Subset(dataset1, range(101))
    test_subset = Subset(dataset2,range(30))
    
    train_images, train_labels = dataset1[0]
    test_images, test_labels = dataset2[0]
    
    rand_params = np.random.normal(2 * np.pi, size=(n_layers, 3))
    
    model = QCNN(train_subset,rand_params).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    test_losses, test_accuracies = [], []
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_subset, optimizer, epoch)
        test_loss, test_acc  = test(model, device, test_subset)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        scheduler.step()
        
    return test_losses, test_accuracies, len(train_subset), len(test_subset)

if __name__ == '__main__':
    main()
