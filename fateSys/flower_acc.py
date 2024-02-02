from collections import OrderedDict
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)
disable_progress_bar()

num_clients = 10
batch_size = 32


def load_datasets():
    fds = FederatedDataset(dataset='cifar10', partitioners={'train': num_clients})

    def apply_transforms(batch):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    trainloaders = []
    valloaders = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=batch_size))

    testset = fds.load_full("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader


trainloaders, valloaders, testloader = load_datasets()

batch = next(iter(trainloaders[0]))

images, labels = batch["img"], batch["label"]

images = images.permute(0, 2, 3, 1).numpy()

images = images / 2 + 0.5
fig, axs = plt.subplots(4, 8, figsize=(12, 6))

for i, ax in enumerate(axs.flat):
    ax.imshow(images[i])
    ax.set_title(trainloaders[0].dataset.features["lables"].in2str(labels[i])[0])
    ax.axis("off")

fig.tight_layout()
plt.show()


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = self.pool1(F.relu(self.conv1(x)))
        content = self.pool1(F.relu(self.conv2(content)))
        content = content.view(-1, 16 * 5 * 5)
        content = F.relu(self.fc1(content))
        content = F.relu(self.fc2(content))
        content = self.fc3(content)
        return content


def train(net, trainloader, epochs: int, verbose=False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for batch in trainloader:
            images, label = batch["img"].to(DEVICE), batch['label'].to(DEVICE)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += label.size(0)
            correct += (torch.max(output.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, label = batch["img"].to(DEVICE), batch['label'].to(DEVICE)
            outputs = net(images)
            loss += criterion(images, label)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


trainloader = trainloaders[0]
valloader = valloaders[0]
net = Net().to(DEVICE)
for epoch in range(5):
    train(net, trainloader, 1)
    loss, accuracy = test(test, valloader)
    print(f"Epoch {epoch + 1}: validation loss {loss}, accuracy {accuracy}")

loss, accuracy = test(net, testloader)
print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader).to_client()


strategy = fl.server.strategy.FedAvg(
    fractions_fit=1.0,
    fractions_evaluate=0.5,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=10,
)

client_resources = {'num_cpus':1, "num_gpus": 0.0}

if DEVICE.type == "cuda":
    client_resources = {"num_cpus":1, "num_gpus":1.0}

fl.simulation.stat_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    configparser=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)


def weighted_average(metrics:List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples,_ in metrics]
    return {'accuracy': sum(accuracies) / sum(examples)}

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.5,
    min_fit_clients=10,
    min_evaluate_clients=5,
    min_available_clients=10,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=num_clients,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    client_resources=client_resources,
)

