import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForImageClassification
from tqdm import tqdm
import datetime
from torch.optim import AdamW
import numpy as np

# try sklearn f1_score
try:
    from sklearn.metrics import f1_score
    SKLEARN_F1 = True
except Exception:
    SKLEARN_F1 = False


class CustomSegformerDataset(Dataset):
    def __init__(self, root_dir, feature_extractor=None, mode="train"):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.images = []
        self.labels = []
        self.mode = mode
        datafolder = os.path.join(self.root_dir, self.mode)

        if not os.path.isdir(datafolder):
            raise FileNotFoundError(f"Data Folder not found at: {datafolder}")
        # paths & labels
        for label_name in sorted(os.listdir(datafolder)):
            class_path = os.path.join(datafolder, label_name)
            if not os.path.isdir(class_path):
                continue
            try:
                class_idx = int(label_name)
            except ValueError:
                continue
            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(os.path.join(class_path, img_name))
                    self.labels.append(class_idx)
        assert len(self.images) == len(self.labels), "Number of images and labels must match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.feature_extractor is None:
            raise ValueError("feature_extractor is None. Provide a SegformerImageProcessor instance.")

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]  # (3,H,W)
        return {"pixel_values": pixel_values, "labels": torch.tensor(label, dtype=torch.long)}


# Prepare Dataloaders
def get_dataloaders(data_root, feature_extractor, batch_size=32, num_workers=4):
    train_dataset = CustomSegformerDataset(data_root, feature_extractor, mode="train")
    val_dataset = CustomSegformerDataset(data_root, feature_extractor, mode="val")
    test_dataset = CustomSegformerDataset(data_root, feature_extractor, mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# Training 
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    if len(dataloader) == 0:
        return 0.0
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()                         # clear grads first
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def _macro_f1_from_confusion(cm):
    # cm is confusion matrix: rows=true, cols=pred
    with np.errstate(divide='ignore', invalid='ignore'):
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0).astype(float) - tp
        fn = cm.sum(axis=1).astype(float) - tp
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1_per_class = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    # macro average across classes
    return float(np.mean(f1_per_class))


def evaluate(model, dataloader, device):
    """
    Returns: (accuracy, f1_macro)
    """
    import numpy as np
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            labels = inputs["labels"].cpu().numpy()
            preds_all.append(preds)
            labels_all.append(labels)

    if len(preds_all) == 0:
        return 0.0, 0.0
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    acc = float((preds_all == labels_all).mean())

    # compute macro F1
    if SKLEARN_F1:
        f1 = float(f1_score(labels_all, preds_all, average="macro"))
    else:
        # fall back to confusion matrix-based macro-f1
        from collections import defaultdict
        unique_labels = np.unique(np.concatenate([labels_all, preds_all]))
        label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
        cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for t, p in zip(labels_all, preds_all):
            cm[label_to_index[t], label_to_index[p]] += 1
        f1 = _macro_f1_from_confusion(cm)
    return acc, f1


def main(data_root=r"D:/Inteview/Dataset", num_epochs=25, batch_size=32, lr=5e-5, num_workers=4, exp_no="2"):
    
    import numpy as np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create timestamped run folder for logs and model
    run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("logs", exp_no)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "training_log.txt")

    # write header (include Val F1 column)
    with open(log_file, "w") as f:
        f.write("SegFormer Image Classification Training Log\n")
        f.write(f"Start Time: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Dataset: {data_root}\n")
        f.write(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {lr}, Num Workers: {num_workers}\n")
        f.write("=" * 140 + "\n")
        # header for epoch lines
        f.write("Timestamp, Epoch, Train Loss, Val Acc, Val F1_macro, Best Val Acc, Best Epoch, Duration_s\n")
        f.write("=" * 140 + "\n")

    # Initialize processor
    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/mit-b5")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(data_root, feature_extractor, batch_size, num_workers=num_workers)

    # small safety checks and log dataset sizes
    with open(log_file, "a") as f:
        f.write(f"Train samples: {len(train_loader.dataset)}\n")
        f.write(f"Val   samples: {len(val_loader.dataset)}\n")
        f.write(f"Test  samples: {len(test_loader.dataset)}\n")
        f.write("-" * 140 + "\n")

    # Load model
    model = SegformerForImageClassification.from_pretrained(
        "nvidia/mit-b5", num_labels=5, ignore_mismatched_sizes=True).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)

    # trackers for best model (based on val_acc)
    best_val_acc = -1.0
    best_epoch = -1
    best_model_path = os.path.join(run_dir, "best_model.pth")

    # Training loop
    for epoch in range(1, num_epochs + 1):
        start_epoch = datetime.datetime.now()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        end_epoch = datetime.datetime.now()
        duration_s = (end_epoch - start_epoch).total_seconds()

        # check for improvement -> save best model (based on val_acc)
        improved = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # save best model checkpoint with val_f1 included
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_f1": val_f1,
                "train_loss": train_loss,
            }, best_model_path)
            improved = True

        # build log line including Val F1
        timestamp = datetime.datetime.now().isoformat()
        log_line = f"Epoch [{epoch}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Best Val: {best_val_acc:.4f} (epoch {best_epoch}) | Duration: {duration_s:.1f}s" + ("  <-- improved" if improved else "")
        print(log_line)

        # Append to log file
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    # Final test accuracy + f1
    test_acc, test_f1 = evaluate(model, test_loader, device)
    test_line = f"Final Test Accuracy: {test_acc:.4f}, Test F1_macro: {test_f1:.4f}"
    print(test_line)
    with open(log_file, "a") as f:
        f.write("=" * 140 + "\n")
        f.write(f"{datetime.datetime.now().isoformat()} - {test_line}\n")

    # Save final model into run folder (last epoch)
    final_model_path = os.path.join(run_dir, "final_model.pth")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_acc": test_acc,
        "test_f1": test_f1,
    }, final_model_path)
    msg = f"Final model saved as {final_model_path}"
    print(msg)
    with open(log_file, "a") as f:
        f.write(msg + "\n")
        f.write(f"End Time: {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 140 + "\n")

    # Summary print
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    if os.path.exists(best_model_path):
        print(f"Best model saved to: {os.path.abspath(best_model_path)}")
    print(f"Training log saved to {os.path.abspath(log_file)}")
    print(f"Run directory: {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
