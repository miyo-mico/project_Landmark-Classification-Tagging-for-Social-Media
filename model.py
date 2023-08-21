import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), # Output: (16, 224, 224),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (16, 112, 112)
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 112, 112)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 56, 56)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (64, 56, 56)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, 28, 28),
            
            nn.Conv2d(64, 128, 3, padding=1), # Output: (128, 28, 28)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2,2), # Output: (128, 14, 14)
            nn.ReLU(),
            
            nn.Conv2d(128, 256, 3, padding=1), # Output: (256, 14, 14)
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2,2), # Output: (256, 7, 7)
            nn.ReLU(),
            
            nn.Flatten(),
        
            nn.Linear(256*7*7, 3000),
            nn.BatchNorm1d(3000),
            nn.ReLU(),
            
            nn.Linear(3000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(in_features=1000, out_features=num_classes)
        )

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return x


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
