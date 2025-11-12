import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForImageClassification, SegformerImageProcessor
from PIL import Image
import math

class GradCamSegformer:
    def __init__(self, model, processor, target_layer_name=None, device='cuda'):
        """
        model: SegformerForImageClassification (PyTorch)
        target_layer_name: name of module to hook; if None, we try to use last backbone output from hidden_states
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.model.to(device).eval()
        self.feature_maps = None
        self.gradients = None
        self.hook_handles = []

        # We'll try to hook the last mix-vision transformer output if possible.
        # Many SegFormer wrappers expose backbone.mix_output or return hidden_states; we use hidden_states approach below.
        # If you want to hook a named module, you can pass its name and register hooks.

    def _register_hooks_module(self, module):
        def forward_hook(module, input, output):
            # output shape expected (batch, seq_len, dim) or (batch, C, H, W)
            self.feature_maps = output.detach()
        def backward_hook(module, grad_in, grad_out):
            # grad_out is a tuple
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(module.register_forward_hook(forward_hook))
        self.hook_handles.append(module.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

    def generate(self, image: Image.Image, target_class=None):
        """
        image: PIL image
        target_class: int class index; if None, use predicted class
        returns: heatmap upsampled to original image size (H, W) numpy array normalized 0..1
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # request hidden states so we can pick backbone outputs
        out = self.model(**inputs, output_hidden_states=True, return_dict=True)
        logits = out.logits  # (1, num_labels)
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        # Choose last hidden state from backbone (often hidden_states[-1])
        # hidden_states is a tuple: (embedding, layer1, ..., layerN, classifier) depending on config
        # For SegFormer, hidden_states[-1] is likely last encoder output (batch, seq_len, dim)
        last_hidden = out.hidden_states[-1]  # (1, seq_len, dim)
        # Save a copy and require grad by recomputing logits from that feature (call through classifier)
        # But we need to compute gradient of logits[target_class] w.r.t last_hidden.
        # We'll re-run forward with hooks: easier to get a tensor we can call backward on.
        # Convert last_hidden into shape (B, C, H, W)
        seq_len = last_hidden.shape[1]
        dim = last_hidden.shape[2]
        grid_size = int(math.sqrt(seq_len))
        if grid_size * grid_size != seq_len:
            # handle possible special token at start
            if seq_len - 1 == int(math.sqrt(seq_len - 1))**2:
                last_hidden = last_hidden[:, 1:, :]
                seq_len = last_hidden.shape[1]
                grid_size = int(math.sqrt(seq_len))
            else:
                raise ValueError("Cannot reshape last hidden tokens into square grid. seq_len={}".format(seq_len))

        fmap = last_hidden.permute(0,2,1).reshape(1, dim, grid_size, grid_size)  # (B, C, H, W)
        fmap = fmap.requires_grad_(True)

        # Create a new classifier head: use model.classifier or model.head - different HF implementations vary.
        # We'll pass fmap through global pooling and classifier weights manually if possible.
        # Many SegFormer classification heads expect flattened features; easiest route: use original model's classifier
        # but that expects inputs from feature extractor pipeline. We'll approximate: global pool fmap -> linear head.

        # Simple approach: re-use model.classifier if available and matches dims
        # Try to find a linear head: model.classifier or model.classifier_norm + model.classifier
        head = None
        if hasattr(self.model, "classifier"):
            head = self.model.classifier
        elif hasattr(self.model, "pre_logits"):  # some variants
            head = self.model.pre_logits
        else:
            # fallback: compute logits by projecting fmap to the same dimension as model.classifier weight if exists
            pass

        # Compute pooled feature (global average) to get a (1, dim) vector
        pooled = fmap.mean(dim=[2,3])  # (1, dim)
        # If head exists and is Linear, pass through it:
        try:
            if head is not None:
                logits_from_fmap = head(pooled)
            else:
                # fallback: use model.classifier from original outputs by replacing hidden_states[-1] with fmap flattened
                # This is model-specific; if this fails, user should inspect model.head architecture.
                logits_from_fmap = self.model.classifier(pooled)
        except Exception:
            # last-resort: use original logits and compute gradients via hooks:
            # compute scalar = logits[0, target_class] and backprop since last_hidden contributed to logits in original forward
            # This requires last_hidden to be attached to computation graph; earlier we detached when calling model once.
            # So re-run forward while capturing last_hidden as a module output is more robust.
            # Simpler: recompute full forward with inputs, but set hooks to capture features and gradients.
            # We'll do that instead of brittle fallback.
            self.remove_hooks()
            # find a module to hook: many SegFormer models have backbone.encoder or model.segformer.encoder... we'll try common names:
            candidate_names = ["segformer.encoder", "model.encoder", "encoder", "backbone", "mix_encoder"]
            hooked = False
            for name, module in self.model.named_modules():
                if "mix" in name and "layer" in name:
                    # try hooking the parent
                    self._register_hooks_module(module)
                    hooked = True
                    break
            if not hooked:
                # as last resort hook the whole model
                self._register_hooks_module(self.model)
            # Now rerun forward and backward
            out2 = self.model(**inputs, output_hidden_states=False, output_attentions=False, return_dict=True)
            logits2 = out2.logits
            score = logits2[0, target_class]
            self.model.zero_grad()
            score.backward(retain_graph=False)
            # now try to retrieve feature_maps and gradients from hooks
            if self.feature_maps is None or self.gradients is None:
                raise RuntimeError("Failed to capture feature maps and gradients with hooks. Consider specifying a target_layer_name.")
            # feature_maps shape might be (1, seq_len, dim) or (1, C, H, W). Normalize to (B,C,H,W)
            fmap_hook = self.feature_maps
            grad_hook = self.gradients
            if fmap_hook.ndim == 3:  # (B, seq_len, dim)
                seq_len = fmap_hook.shape[1]
                dim = fmap_hook.shape[2]
                grid = int(math.sqrt(seq_len))
                fmap_hook = fmap_hook.permute(0,2,1).reshape(1, dim, grid, grid)
            if grad_hook.ndim == 3:
                grad_hook = grad_hook.permute(0,2,1).reshape(1, dim, grid, grid)

            weights = grad_hook.mean(dim=(2,3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * fmap_hook).sum(dim=1, keepdim=True)  # (1,1,H,W)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=image.size[::-1], mode='bilinear', align_corners=False)
            cam = cam.squeeze().detach().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            self.remove_hooks()
            return cam

        # If we reached here, logits_from_fmap exists and is differentiable w.r.t fmap
        score = logits_from_fmap[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # Now pooled.grad or fmap.grad may have gradients
        if fmap.grad is None:
            raise RuntimeError("No gradients captured on fmap. Ensure fmap.requires_grad_(True) and head uses pooled features.")

        gradients = fmap.grad[0]  # (C, H, W)
        activations = fmap.detach()[0]  # (C, H, W)
        # Global-average pooling of gradients for weights
        weights = gradients.mean(dim=(1,2))  # (C,)
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = torch.relu(cam)
        # Upsample to image size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        cam_up = F.interpolate(cam, size=image.size[::-1], mode='bilinear', align_corners=False)
        cam_up = cam_up.squeeze().cpu().numpy()
        cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
        return cam_up

# Example usage:
img_path =r"D:/Inteview/Dataset/test/4/9886298L.png"
model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
gc = GradCamSegformer(model, processor, device='cuda')
img = Image.open(img_path).convert("RGB")
heatmap = gc.generate(img)  # normalized 0..1
plt.imshow(img); plt.imshow(heatmap, cmap='jet', alpha=0.45); plt.axis('off'); plt.show()
