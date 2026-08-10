"""Dataset loading utilities."""

from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloader(
    data_dir: str, img_size: int, batch_size: int
) -> Tuple[DataLoader, List[str], int]:
    """Build an ImageFolder dataloader normalized to ``[-1, 1]``.

    Expects an ImageFolder layout: ``data_dir/class_A/``, ``data_dir/class_B/``.
    Images are normalized to ``[-1, 1]`` to match the generator's tanh output.

    Returns the dataloader, the list of class names, and the class count.
    """
    # No RandomHorizontalFlip here: the ADA pipeline already applies flips, so
    # doing it twice double-augments the reals only (not the fakes).
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    num_classes = len(class_names)

    print(f"Dataset: {data_dir}")
    print(f"  Classes: {class_names} | Total images: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
        drop_last=True,  # needed for MinibatchStdDev
    )

    return dataloader, class_names, num_classes
