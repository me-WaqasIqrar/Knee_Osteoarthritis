import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForImageClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import datetime
import random
# Optional sklearn import for better confusion matrix plotting & f1 score
try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class CustomSegformerDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, feature_extractor=None, mode="test"):
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.images = []
        self.labels = []
        self.mode = mode

        datafolder = os.path.join(self.root_dir, self.mode)
        if not os.path.isdir(datafolder):
            raise FileNotFoundError(f"Data folder not found at: {datafolder}")

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
            raise ValueError("Feature extractor is None. Provide SegformerImageProcessor.")

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": torch.tensor(label), "img_path": img_path}


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    preds_all, labels_all = [], []
    for batch in tqdm(dataloader, desc="Evaluating"):
        inputs = {k: v.to(device) for k, v in batch.items() if k in ("pixel_values", "labels")}
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        labels = inputs["labels"].cpu().numpy()
        preds_all.append(preds)
        labels_all.append(labels)
    if len(preds_all) == 0:
        return np.array([]), np.array([])
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    return preds_all, labels_all


def save_confusion_matrix(y_true, y_pred, save_dir, labels=None):
    os.makedirs(save_dir, exist_ok=True)
    cm_png = os.path.join(save_dir, "confusion_matrix.png")
    cm_csv = os.path.join(save_dir, "confusion_matrix.csv")

    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(values_format="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(cm_png)
        plt.close()
    else:
        # simple fallback
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        cm = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        np.savetxt(cm_csv, cm, fmt="%d", delimiter=",")
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.xticks(range(len(unique_labels)), unique_labels)
        plt.yticks(range(len(unique_labels)), unique_labels)
        plt.tight_layout()
        plt.savefig(cm_png)
        plt.close()

    return cm_png, cm_csv


def macro_f1_numpy(y_true, y_pred):
    """Compute macro F1 from arrays using confusion matrix (safe fallback)."""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    # compute per-class precision, recall, f1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
        f1_per_class = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) != 0)
    # macro average
    return float(np.mean(f1_per_class))


def infer_examples(model, dataset, device, n_samples=10):
    print("\nSample Inference Results:")
    model.eval()
    total = len(dataset)
    if total == 0:
        print("No samples in dataset.")
        return
    n = min(n_samples, total)
    indices = random.sample(range(total), k=n)
    for i in indices:
        item = dataset[i]
        inputs = {"pixel_values": item["pixel_values"].unsqueeze(0).to(device)}
        with torch.no_grad():
            outputs = model(**inputs)
            pred = int(outputs.logits.argmax(dim=-1).cpu().item())
        true_label = int(item["labels"].item())
        img_name = os.path.basename(item["img_path"])
        print(f"{img_name:<30} | Original: {true_label} -> Predicted: {pred}")


def main():
    # === MODIFY THESE PATHS AND SETTINGS ===
    data_root = r"D:/Knee_Osteoarthritis/Dataset"      # dataset root (contains train/val/test)
    model_path = r"logs/2/best_model.pth"  # path to saved model
    backbone = "nvidia/mit-b5"
    num_labels = 5
    batch_size = 16
    num_workers = 4
    n_examples = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize processor and dataset
    feature_extractor = SegformerImageProcessor.from_pretrained(backbone)
    test_dataset = CustomSegformerDataset(data_root, feature_extractor, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model
    model = SegformerForImageClassification.from_pretrained(backbone, num_labels=num_labels, ignore_mismatched_sizes=True)
    state = torch.load(model_path, map_location="cpu")
    # support both state_dict or saved dict with "model_state_dict"
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Evaluate on test set
    preds, labels = evaluate_model(model, test_loader, device)
    if preds.size == 0:
        print("No predictions (empty test set). Exiting.")
        return

    # Accuracy
    if SKLEARN_AVAILABLE:
        acc = float(accuracy_score(labels, preds))
    else:
        acc = float((preds == labels).mean())

    # F1 (macro)
    if SKLEARN_AVAILABLE:
        f1 = float(f1_score(labels, preds, average="macro"))
    else:
        f1 = macro_f1_numpy(labels, preds)

    print(f"\nTest Accuracy: {acc:.4f} ({len(labels)} samples)")
    print(f"Test F1 (macro): {f1:.4f}")

    # Save confusion matrix
    save_dir = os.path.dirname(os.path.abspath(model_path))
    cm_png, cm_csv = save_confusion_matrix(labels, preds, save_dir)
    print(f"Confusion matrix saved to:\n {cm_png}\n {cm_csv}")

    # Save inference run scores to text file
    score_file = os.path.join(save_dir, "inference_run_score.txt")
    with open(score_file, "w") as f:
        f.write("Inference run results\n")
        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"Model path: {os.path.abspath(model_path)}\n")
        f.write(f"Dataset (test folder): {os.path.abspath(os.path.join(data_root, 'test'))}\n")
        f.write(f"Num samples: {len(labels)}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"F1_macro: {f1:.6f}\n")
    print(f"Inference scores saved to: {score_file}")

    # Show sample inferences
    infer_examples(model, test_dataset, device, n_samples=n_examples)


if __name__ == "__main__":
    main()
