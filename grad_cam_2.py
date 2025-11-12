import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerForImageClassification, SegformerImageProcessor, SegformerConfig

# ----------------------------- USER CONFIG -----------------------------
CHECKPOINT_PATH = r"D:/Inteview/logs/1/final_model.pth"
path=r"D:/Inteview/Dataset/test"
IMG_PATH =r"D:/Inteview/Dataset/test/2/9797850R.png"
HF_BACKBONE = "nvidia/mit-b0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULT_SAVE_PATH = r"D:/Inteview/results/2"
OUT_SAVE = True
OVERLAY_ALPHA = 0.45
OUT_FILENAME = None
USE_WEIGHTED_LAYERS = True  # weight deeper layers stronger when averaging
USE_GRADCAM_FALLBACK = True  # try Grad-CAM if attentions aren't usable
# -----------------------------------------------------------------------



def infer_num_labels_from_checkpoint(ck):
    """Try to infer num_labels from checkpoint classifier weight/bias shapes."""
    if not isinstance(ck, dict):
        return None
    # search for keys that look like classifier.weight
    candidates = []
    for k, v in ck.items():
        if k.endswith("classifier.weight") and hasattr(v, "shape"):
            candidates.append((k, v.shape[0]))
        if k.endswith("classifier.bias") and hasattr(v, "shape"):
            candidates.append((k, v.shape[0]))
    if candidates:
        # return the most common value
        vals = [c[1] for c in candidates]
        return int(max(set(vals), key=vals.count))
    return None


def load_model_from_checkpoint(checkpoint_path, hf_backbone, device):
    print(f"Loading HF config ({hf_backbone})...")
    # Try to read checkpoint first to detect num_labels
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ck = torch.load(checkpoint_path, map_location='cpu')
    inferred = infer_num_labels_from_checkpoint(ck)

    cfg = SegformerConfig.from_pretrained(hf_backbone)
    if inferred is not None:
        print(f"Inferred num_labels={inferred} from checkpoint. Setting config.num_labels accordingly.")
        cfg.num_labels = inferred
    else:
        print("Could not infer num_labels from checkpoint; using default config.num_labels.")

    model = SegformerForImageClassification.from_pretrained(hf_backbone, config=cfg)
    model.to(device)

    # extract plausible state_dict
    if isinstance(ck, dict) and "state_dict" in ck:
        state_dict = ck["state_dict"]
    elif isinstance(ck, dict) and any(isinstance(v, dict) for v in ck.values()):
        # choose the largest dict value heuristic
        candidate = None
        for k, v in ck.items():
            if isinstance(v, dict):
                if candidate is None or len(v) > len(candidate):
                    candidate = v
        state_dict = candidate if candidate is not None else ck
    else:
        state_dict = ck

    # try strict load then fallback
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Loaded checkpoint (strict=True).")
        return model
    except Exception as e_strict:
        print("Strict load failed, trying non-strict load. Error:\n", e_strict)

    # Strip common prefixes
    new_state = {}
    for k, v in state_dict.items():
        new_k = k
        if k.startswith("module."):
            new_k = new_k[len("module."):]
        if k.startswith("model."):
            new_k = new_k[len("model."):]
        new_state[new_k] = v

    try:
        model.load_state_dict(new_state, strict=False)
        print("Loaded checkpoint with strict=False (prefix-stripped). Some keys may have been skipped.")
    except Exception as e2:
        print("Non-strict load also failed. Attempting selective-key loading. Error:\n", e2)
        compatible_state = {}
        model_keys = model.state_dict()
        for k, v in new_state.items():
            if k in model_keys and model_keys[k].shape == v.shape:
                compatible_state[k] = v
        missing = set(model_keys.keys()) - set(compatible_state.keys())
        model_keys.update(compatible_state)
        model.load_state_dict(model_keys)
        print(f"Loaded {len(compatible_state)} matching keys into model state dict. {len(missing)} keys left at default.")

    return model


def visualize_attention_multiscale(model, processor, image: Image.Image, device='cuda', weight_layers=False):
    model.to(device).eval()
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs, output_attentions=True, return_dict=True)

    attentions = outputs.attentions
    if attentions is None or len(attentions) == 0:
        raise RuntimeError("No attentions returned. Consider using Grad-CAM fallback.")

    img_w, img_h = image.size
    layer_maps = []
    layer_weights = []

    for i, attn in enumerate(attentions):
        # attn: (batch, heads, q, k)
        a = attn.mean(dim=1)  # (batch, q, k)
        key_importance = a.mean(dim=1)  # (batch, k)
        key_vec = key_importance[0].detach().cpu().numpy()

        L = key_vec.shape[0]
        g = int(math.sqrt(L))
        if g * g != L:
            # drop leading special token if present
            if (L - 1) == int(math.sqrt(L - 1))**2:
                key_vec = key_vec[1:]
                L = key_vec.shape[0]
                g = int(math.sqrt(L))
            else:
                if L == 1:
                    heat_small = np.ones((1,1), dtype=np.float32) * float(key_vec[0])
                    heat_t = torch.tensor(heat_small, device=device).unsqueeze(0).unsqueeze(0).float()
                    heat_up = F.interpolate(heat_t, size=(img_h, img_w), mode='bilinear', align_corners=False)
                    heat_up = heat_up.squeeze().cpu().numpy()
                    layer_maps.append(heat_up)
                    layer_weights.append(i+1 if weight_layers else 1.0)
                    continue
                # pad to nearest square
                sq = int(np.ceil(np.sqrt(L)))
                padded = np.zeros(sq * sq, dtype=key_vec.dtype)
                padded[:L] = key_vec
                key_vec = padded
                g = sq

        heat_small = key_vec.reshape(g, g)
        heat_t = torch.tensor(heat_small, device=device).unsqueeze(0).unsqueeze(0).float()
        heat_up = F.interpolate(heat_t, size=(img_h, img_w), mode='bilinear', align_corners=False)
        heat_up = heat_up.squeeze().cpu().numpy()
        heat_up = (heat_up - heat_up.min()) / (heat_up.max() - heat_up.min() + 1e-8)
        layer_maps.append(heat_up)
        layer_weights.append(i+1 if weight_layers else 1.0)

    weights = np.array(layer_weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-12)
    stacked = np.stack(layer_maps, axis=0)
    heatmap_up = (weights[:, None, None] * stacked).sum(axis=0)
    heatmap_up = (heatmap_up - heatmap_up.min()) / (heatmap_up.max() - heatmap_up.min() + 1e-8)
    return heatmap_up


class GradCamSegformer:
    def __init__(self, model, processor, device='cuda'):
        self.model = model
        self.processor = processor
        self.device = device
        self.feature_maps = None
        self.gradients = None
        self.hook_handles = []
        self.model.to(device).eval()

    def _register_forward_backward_hooks(self, module):
        def forward_hook(module, inp, out):
            # out may be (B, seq, dim)
            self.feature_maps = out
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.hook_handles.append(module.register_forward_hook(forward_hook))
        # backward_hook is deprecated in some PyTorch versions, but we'll try
        try:
            self.hook_handles.append(module.register_backward_hook(backward_hook))
        except Exception:
            pass

    def remove_hooks(self):
        for h in self.hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self.hook_handles = []

    def generate(self, image: Image.Image, target_class=None):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # try to hook a sensible module: many HF Segformer models have `segformer.encoder` or just `encoder`
        hooked = False
        for name, module in self.model.named_modules():
            # pick a module that likely returns hidden tokens (heuristic)
            if "encoder" in name and len(list(module.children()))>0:
                self._register_forward_backward_hooks(module)
                hooked = True
                break
        if not hooked:
            # fall back to hooking model (less ideal)
            self._register_forward_backward_hooks(self.model)

        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        logits = out.logits
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.feature_maps is None or self.gradients is None:
            self.remove_hooks()
            raise RuntimeError("Could not capture feature maps or gradients for Grad-CAM. Try specifying hook module manually.")

        fmap = self.feature_maps
        grad = self.gradients
        # reshape if necessary
        if fmap.ndim == 3:
            # (B, seq, dim)
            seq = fmap.shape[1]; dim = fmap.shape[2]
            g = int(math.sqrt(seq))
            if g*g == seq:
                fmap = fmap.permute(0,2,1).reshape(1, dim, g, g)
                grad = grad.permute(0,2,1).reshape(1, dim, g, g)
            else:
                # drop first token if it helps
                if seq-1 == int(math.sqrt(seq-1))**2:
                    fmap = fmap[:,1:,:]
                    grad = grad[:,1:,:]
                    seq = fmap.shape[1]
                    g = int(math.sqrt(seq))
                    fmap = fmap.permute(0,2,1).reshape(1, dim, g, g)
                    grad = grad.permute(0,2,1).reshape(1, dim, g, g)
                else:
                    self.remove_hooks()
                    raise RuntimeError("Grad-CAM: cannot reshape feature tokens to spatial map.")

        weights = grad.mean(dim=(2,3), keepdim=True)
        cam = (weights * fmap).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.size[::-1], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        self.remove_hooks()
        return cam


def overlay_and_save(image: Image.Image, heatmap: np.ndarray, alpha=0.45, out_path=None):
    plt.figure(figsize=(6,6), dpi=200)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.axis('off')
    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved overlay to: {out_path}")
    else:
        plt.show()
    plt.close()

def main_inference(model, processor, IMG_PATH,label,save_dir):
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    img = Image.open(IMG_PATH).convert('RGB')

    print(f"Attempting multiscale attention visualization on {IMG_PATH} using device={DEVICE}...")
    try:
        heatmap = visualize_attention_multiscale(model, processor, img, device=DEVICE, weight_layers=USE_WEIGHTED_LAYERS)
        print("Attention map generated.")
    except Exception as e:
        print("Attention visualization failed: ", e)
        heatmap = None

    if heatmap is None and USE_GRADCAM_FALLBACK:
        try:
            print("Trying Grad-CAM fallback...")
            gc = GradCamSegformer(model, processor, device=DEVICE)
            heatmap = gc.generate(img)
            print("Grad-CAM map generated.")
        except Exception as e2:
            print("Grad-CAM also failed: ", e2)
            raise RuntimeError("Both attention and Grad-CAM visualization failed; please inspect model internals.")

    if OUT_FILENAME is None:
        if not os.path.exists(RESULT_SAVE_PATH):
            os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
        out_name = os.path.join(RESULT_SAVE_PATH, "attention_overlay.png")
    else:
        out_name = OUT_FILENAME

    if OUT_SAVE:
        overlay_and_save(img, heatmap, alpha=OVERLAY_ALPHA, out_path=out_name)
    else:
        overlay_and_save(img, heatmap, alpha=OVERLAY_ALPHA, out_path=None)

    print("Done.")


if __name__ == '__main__':
    model = load_model_from_checkpoint(CHECKPOINT_PATH, HF_BACKBONE, DEVICE)
    processor = SegformerImageProcessor.from_pretrained(HF_BACKBONE)
    main_inference(model, processor, IMG_PATH)