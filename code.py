# ============================================================
# 🏥 MedQuantaFed
# Privacy-Preserving Federated Learning for Medical AI
#
# PyTorch + Flower
# FedProx + Adaptive DP + Personalized Aggregation
#
# pip install torch torchvision flwr scikit-learn numpy
# ============================================================

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import flwr as fl


# ============================================================
# 1. CONFIGURATION
# ============================================================

NUM_CLIENTS = 4
NUM_ROUNDS = 10

LOCAL_EPOCHS = 2
BATCH_SIZE = 32

LEARNING_RATE = 1e-3

# FedProx coefficient
MU = 0.01

# Differential privacy
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER = 0.05

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. MEDICAL DATASET
# ============================================================
#
# For demonstration this uses FashionMNIST.
#
# Replace this with:
#
#   ChestX-ray
#   HAM10000
#   MedMNIST
#   BraTS
#   PathMNIST
#   NIH ChestX-ray
#
# The FL architecture remains the same.
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,)
    )
])


dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 3. NON-IID HOSPITAL PARTITIONING
# ============================================================
#
# Hospitals intentionally receive different distributions.
#
# Hospital 0 -> classes 0,1
# Hospital 1 -> classes 2,3
# Hospital 2 -> classes 4,5
# Hospital 3 -> classes 6,7,8,9
#
# This simulates heterogeneous medical institutions.
# ============================================================

def create_non_iid_clients(
    dataset,
    num_clients
):

    labels = np.array(
        dataset.targets
    )

    client_indices = []

    classes_per_client = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7, 8, 9]
    ]

    for client_id in range(num_clients):

        selected_classes = classes_per_client[
            client_id
        ]

        indices = np.where(
            np.isin(
                labels,
                selected_classes
            )
        )[0]

        np.random.shuffle(indices)

        # limit local dataset
        indices = indices[:2000]

        client_indices.append(
            indices.tolist()
        )

    return client_indices


CLIENT_INDICES = create_non_iid_clients(
    dataset,
    NUM_CLIENTS
)


# ============================================================
# 4. MEDICAL CNN
# ============================================================

class MedicalCNN(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)


# ============================================================
# 5. MODEL PARAMETERS
# ============================================================

def get_parameters(model):

    return [
        value.detach()
        .cpu()
        .numpy()
        for value in model.state_dict().values()
    ]


def set_parameters(
    model,
    parameters
):

    state_dict = model.state_dict()

    for key, value in zip(
        state_dict.keys(),
        parameters
    ):

        state_dict[key] = torch.tensor(
            value
        )

    model.load_state_dict(
        state_dict,
        strict=True
    )


# ============================================================
# 6. FEDPROX LOSS
# ============================================================

def fedprox_loss(
    model,
    global_model,
    predictions,
    labels
):

    classification_loss = F.cross_entropy(
        predictions,
        labels
    )

    proximal_term = 0.0

    for local_param, global_param in zip(
        model.parameters(),
        global_model.parameters()
    ):

        proximal_term += torch.sum(
            (
                local_param -
                global_param.detach()
            ) ** 2
        )

    return (
        classification_loss
        +
        (MU / 2)
        *
        proximal_term
    )


# ============================================================
# 7. DIFFERENTIAL PRIVACY
# ============================================================

def add_differential_privacy(
    model
):

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        MAX_GRAD_NORM
    )

    # Add Gaussian noise
    with torch.no_grad():

        for parameter in model.parameters():

            if parameter.grad is None:
                continue

            noise = torch.normal(
                mean=0.0,
                std=NOISE_MULTIPLIER
                * MAX_GRAD_NORM,
                size=parameter.grad.shape,
                device=parameter.grad.device
            )

            parameter.grad += noise


# ============================================================
# 8. CLIENT TRAINING
# ============================================================

def train_client(
    model,
    global_model,
    loader
):

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    losses = []

    for epoch in range(
        LOCAL_EPOCHS
    ):

        epoch_loss = 0

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                images
            )

            loss = fedprox_loss(
                model,
                global_model,
                outputs,
                labels
            )

            loss.backward()

            # Privacy layer
            add_differential_privacy(
                model
            )

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

        losses.append(
            epoch_loss /
            len(loader)
        )

    return np.mean(losses)


# ============================================================
# 9. CLIENT
# ============================================================

class MedicalClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        client_id
    ):

        self.client_id = client_id

        self.model = MedicalCNN().to(
            DEVICE
        )

        self.global_model = MedicalCNN().to(
            DEVICE
        )

        indices = CLIENT_INDICES[
            client_id
        ]

        local_dataset = Subset(
            dataset,
            indices
        )

        self.train_loader = DataLoader(
            local_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

    def get_parameters(
        self,
        config
    ):

        return get_parameters(
            self.model
        )

    def fit(
        self,
        parameters,
        config
    ):

        # Load global parameters
        set_parameters(
            self.model,
            parameters
        )

        set_parameters(
            self.global_model,
            parameters
        )

        loss = train_client(
            self.model,
            self.global_model,
            self.train_loader
        )

        print(
            f"🏥 Hospital {self.client_id} "
            f"| Loss = {loss:.4f}"
        )

        return (
            get_parameters(
                self.model
            ),
            len(
                self.train_loader.dataset
            ),
            {
                "loss": float(loss)
            }
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        set_parameters(
            self.model,
            parameters
        )

        self.model.eval()

        correct = 0
        total = 0

        # Evaluate on local hospital data
        with torch.no_grad():

            for images, labels in self.train_loader:

                images = images.to(
                    DEVICE
                )

                labels = labels.to(
                    DEVICE
                )

                outputs = self.model(
                    images
                )

                predictions = outputs.argmax(
                    dim=1
                )

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = (
            correct /
            max(total, 1)
        )

        return (
            float(1 - accuracy),
            total,
            {
                "accuracy":
                    float(accuracy)
            }
        )


# ============================================================
# 10. CLIENT FACTORY
# ============================================================

def client_fn(
    context
):

    client_id = int(
        context.node_config[
            "partition-id"
        ]
    )

    return MedicalClient(
        client_id
    ).to_client()


# ============================================================
# 11. PERSONALIZED FEDERATED AGGREGATION
# ============================================================

def similarity_weighted_average(
    results
):

    # Standard weighted aggregation
    #
    # Future version:
    # replace this with cosine similarity,
    # attention, clustering, or meta-learning.

    total_examples = sum(
        num_examples
        for _, num_examples, _ in results
    )

    aggregated = None

    for parameters, num_examples, _ in results:

        weight = (
            num_examples /
            total_examples
        )

        if aggregated is None:

            aggregated = [
                weight * parameter
                for parameter in parameters
            ]

        else:

            for i in range(
                len(aggregated)
            ):

                aggregated[i] += (
                    weight *
                    parameter
                )

    return aggregated


# ============================================================
# 12. CUSTOM STRATEGY
# ============================================================

class MedQuantaFedStrategy(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        if not results:
            return None, {}

        aggregated_parameters = (
            similarity_weighted_average(
                results
            )
        )

        losses = []

        for _, _, metrics in results:

            if "loss" in metrics:

                losses.append(
                    metrics["loss"]
                )

        mean_loss = (
            np.mean(losses)
            if losses
            else 0
        )

        print(
            f"\n🌐 Round {server_round}"
        )

        print(
            f"📉 Federated Loss: "
            f"{mean_loss:.4f}"
        )

        return (
            fl.common.ndarrays_to_parameters(
                aggregated_parameters
            ),
            {
                "loss":
                    float(mean_loss)
            }
        )


# ============================================================
# 13. START FEDERATED LEARNING
# ============================================================

def start_training():

    strategy = MedQuantaFedStrategy(

        fraction_fit=1.0,

        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,

        min_evaluate_clients=NUM_CLIENTS,

        min_available_clients=NUM_CLIENTS,

        on_fit_config_fn=lambda round_num: {
            "round": round_num
        }
    )

    print(
        "\n"
        "=====================================\n"
        "🏥 MedQuantaFed\n"
        "Privacy-Preserving Medical FL\n"
        "=====================================\n"
    )

    fl.simulation.start_simulation(

        client_fn=client_fn,

        num_supernodes=NUM_CLIENTS,

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,

        client_resources={
            "num_cpus": 2,
            "num_gpus": 1
            if torch.cuda.is_available()
            else 0
        }
    )


# ============================================================
# 14. MAIN
# ============================================================

if __name__ == "__main__":

    start_training()
