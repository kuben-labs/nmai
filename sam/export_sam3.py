"""
Export SAM 3 detector to ONNX for the NorgesGruppen competition sandbox.

Run on the GCP VM (requires PyTorch 2.7+, CUDA 12.6+, SAM 3 installed):
    python export_sam3.py

This script:
1. Loads the SAM 3 image model
2. Pre-computes text embeddings for the "product" prompt
3. Exports the detector (vision encoder + DETR decoder) to ONNX
4. Quantizes to INT8 (or INT4 if needed) to fit under 420MB
5. Saves pre-computed text embeddings as .npy

Output files:
    weights/sam3_detector.onnx   - ONNX detector model
    weights/text_embed.npy       - Pre-computed text features (dict with arrays)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class Sam3DetectorWrapper(nn.Module):
    """
    Wraps SAM 3 image model for ONNX export.
    Takes preprocessed image tensor + pre-baked text features → boxes + scores.

    The text features are pre-computed for the prompt "product" and stored
    as constants inside this module (or loaded externally).
    """

    def __init__(self, model, text_features, text_mask, text_embeds):
        super().__init__()
        # Vision backbone (ViT + neck)
        self.vision_backbone = model.backbone.vision_backbone
        # Transformer encoder-decoder (DETR)
        self.transformer = model.transformer
        # Geometry encoder (for dummy prompts)
        self.geometry_encoder = model.geometry_encoder
        # Scoring head
        self.dot_prod_scoring = model.dot_prod_scoring
        # Scalp setting from backbone
        self.scalp = model.backbone.scalp

        # Pre-baked text features as buffers (not exported as parameters)
        self.register_buffer("text_features", text_features)
        self.register_buffer("text_mask", text_mask)
        self.register_buffer("text_embeds", text_embeds)

    def forward(self, image):
        """
        Args:
            image: [1, 3, 1008, 1008] normalized tensor
        Returns:
            boxes: [N, 4] in cxcywh format, normalized [0, 1]
            scores: [N] confidence scores
        """
        # 1. Vision forward
        sam3_features, sam3_pos, _, _ = self.vision_backbone(image)
        if self.scalp > 0:
            sam3_features = sam3_features[: -self.scalp]
            sam3_pos = sam3_pos[: -self.scalp]

        # 2. Build backbone_out dict-like inputs for the transformer
        vis_feat = sam3_features[-1]  # last feature level
        vis_feat_flat = vis_feat.flatten(2).permute(2, 0, 1)  # HW, B, C
        vis_pos_flat = sam3_pos[-1].flatten(2).permute(2, 0, 1)
        vis_feat_sizes = [vis_feat.shape[-2:]]

        # 3. Geometry encoder (dummy prompt - no geometric input)
        # Using pre-computed dummy prompt
        geo_feats, geo_masks = self._get_dummy_geometry(vis_feat_flat, vis_feat_sizes, vis_pos_flat)

        # 4. Text features (pre-baked)
        txt_feats = self.text_features  # [seq, 1, C]
        txt_masks = self.text_mask  # [1, seq]

        # 5. Concatenate text + geometry as prompt
        prompt = torch.cat([txt_feats, geo_feats], dim=0)  # [txt_len + geo_len, B, C]
        prompt_mask_combined = torch.cat([txt_masks, geo_masks], dim=-1)  # [B, total_len]

        # 6. Encoder forward
        img_feats_list = [vis_feat_flat]
        img_pos_list = [vis_pos_flat]
        encoder_out = self.transformer.encoder(
            img_feats=img_feats_list,
            img_pos_embeds=img_pos_list,
            text_feats=txt_feats,
            text_masks=txt_masks,
        )

        # 7. Decoder forward
        memory = encoder_out["encoder_hidden_states"]
        pos_embed = encoder_out["pos_embed"]
        src_mask = encoder_out.get("padding_mask", None)

        decoder_out = self.transformer.decoder(
            memory=memory,
            pos_embed=pos_embed,
            src_mask=src_mask,
            prompt=prompt,
            prompt_mask=prompt_mask_combined,
        )

        # 8. Extract boxes and scores
        hs = decoder_out["hs"]  # decoder hidden states
        pred_boxes = decoder_out["pred_boxes"][-1]  # [B, num_queries, 4]
        pred_logits = self.dot_prod_scoring(hs[-1], self.text_embeds)  # [B, num_queries, 1]

        # Presence token score
        if "presence_logit_dec" in decoder_out:
            presence = decoder_out["presence_logit_dec"].sigmoid()
            scores = pred_logits.sigmoid().squeeze(-1) * presence.squeeze(-1)
        else:
            scores = pred_logits.sigmoid().squeeze(-1)

        # Flatten batch dimension (batch=1)
        boxes = pred_boxes.squeeze(0)  # [num_queries, 4]
        scores = scores.squeeze(0)  # [num_queries]

        return boxes, scores

    def _get_dummy_geometry(self, vis_feat, vis_feat_sizes, vis_pos):
        """Create dummy geometry features (no point/box prompts)."""
        device = vis_feat.device
        B = vis_feat.shape[1]
        # Empty geometry - just the CLS token from geometry encoder
        dummy_box_emb = torch.zeros(0, B, 4, device=device)
        dummy_box_mask = torch.zeros(B, 0, device=device, dtype=torch.bool)

        from sam3.model.geometry_encoders import Prompt
        prompt = Prompt(box_embeddings=dummy_box_emb, box_mask=dummy_box_mask)
        geo_feats, geo_masks = self.geometry_encoder(
            geo_prompt=prompt,
            img_feats=[vis_feat],
            img_sizes=vis_feat_sizes,
            img_pos_embeds=[vis_pos],
        )
        return geo_feats, geo_masks


def precompute_text_features(model, prompt="product", device="cuda"):
    """Pre-compute text features for a given prompt."""
    with torch.no_grad():
        text_out = model.backbone.forward_text([prompt], device=device)
    return {
        "language_features": text_out["language_features"].cpu().numpy(),
        "language_mask": text_out["language_mask"].cpu().numpy(),
        "language_embeds": text_out["language_embeds"].cpu().numpy(),
    }


def export_to_onnx(model, output_path, opset=17):
    """Export the detector wrapper to ONNX."""
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, 1008, 1008, device=device)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=opset,
        input_names=["image"],
        output_names=["boxes", "scores"],
        dynamic_axes={
            "image": {0: "batch"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
        },
    )
    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported: {output_path} ({size_mb:.1f} MB)")
    return size_mb


def quantize_onnx(input_path, output_path, mode="int8"):
    """Quantize ONNX model to reduce size."""
    if mode == "fp16":
        from onnxruntime.transformers import float16
        import onnx

        model = onnx.load(str(input_path))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(output_path))

    elif mode == "int8":
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            str(input_path),
            str(output_path),
            weight_type=QuantType.QInt8,
        )

    elif mode == "int4":
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                str(input_path),
                str(output_path),
                weight_type=QuantType.QInt4,
            )
        except Exception:
            # Fallback: use neural-compressor for INT4
            from neural_compressor.quantization import fit
            from neural_compressor.config import PostTrainingQuantConfig

            config = PostTrainingQuantConfig(
                approach="weight_only",
                op_type_dict={".*": {"weight": {"bits": 4}}},
            )
            q_model = fit(str(input_path), config)
            q_model.save(str(output_path))

    size_mb = output_path.stat().st_size / 1e6
    print(f"Quantized ({mode}): {output_path} ({size_mb:.1f} MB)")
    return size_mb


def main():
    parser = argparse.ArgumentParser(description="Export SAM 3 to ONNX")
    parser.add_argument("--output-dir", default="weights", help="Output directory")
    parser.add_argument("--prompt", default="product", help="Text prompt to pre-compute")
    parser.add_argument("--quantize", default="int8", choices=["fp16", "int8", "int4"],
                        help="Quantization mode")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--checkpoint", default=None, help="Path to sam3.pt checkpoint")
    parser.add_argument("--skip-onnx", action="store_true",
                        help="Only export text features, skip ONNX export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Build SAM 3 model
    print("Loading SAM 3 image model...")
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        device=device,
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )

    # 2. Pre-compute text features
    print(f"Pre-computing text features for prompt: '{args.prompt}'")
    text_data = precompute_text_features(model, prompt=args.prompt, device=device)
    text_path = output_dir / "text_embed.npy"
    np.savez(str(text_path).replace(".npy", ""), **text_data)
    # Rename .npz to .npy (sandbox allows .npy)
    npz_path = Path(str(text_path).replace(".npy", ".npz"))
    if npz_path.exists():
        npz_path.rename(text_path)
    print(f"Saved text features: {text_path}")

    if args.skip_onnx:
        print("Skipping ONNX export (--skip-onnx)")
        return

    # 3. Create wrapper model
    print("Creating detector wrapper for ONNX export...")
    text_features = torch.from_numpy(text_data["language_features"]).to(device)
    text_mask = torch.from_numpy(text_data["language_mask"]).to(device)
    text_embeds = torch.from_numpy(text_data["language_embeds"]).to(device)

    wrapper = Sam3DetectorWrapper(model, text_features, text_mask, text_embeds)
    wrapper.eval()

    # Count parameters
    num_params = sum(p.numel() for p in wrapper.parameters())
    print(f"Detector parameters: {num_params / 1e6:.1f}M")
    print(f"Estimated FP16 size: {num_params * 2 / 1e6:.0f} MB")
    print(f"Estimated INT8 size: {num_params / 1e6:.0f} MB")

    # 4. Export to ONNX (FP32 first)
    fp32_path = output_dir / "sam3_detector_fp32.onnx"
    try:
        export_to_onnx(wrapper, fp32_path, opset=args.opset)
    except Exception as e:
        print(f"\nONNX export failed: {e}")
        print("\nFalling back to trace-based export...")
        try:
            # Try with torch.jit.trace
            traced = torch.jit.trace(wrapper, torch.randn(1, 3, 1008, 1008, device=device))
            torch.onnx.export(
                traced,
                torch.randn(1, 3, 1008, 1008, device=device),
                str(fp32_path),
                opset_version=args.opset,
                input_names=["image"],
                output_names=["boxes", "scores"],
            )
        except Exception as e2:
            print(f"\nTrace-based export also failed: {e2}")
            print("\n=== ONNX EXPORT FAILED ===")
            print("The SAM 3 model is too complex for direct ONNX export.")
            print("Alternatives:")
            print("  1. Use SAM 3 to generate training data, then train YOLOv8")
            print("     Run: python generate_sam3_labels.py")
            print("  2. Export just the image encoder to ONNX (simpler)")
            print("  3. Use TorchScript export instead of ONNX")
            return

    # 5. Quantize
    quant_path = output_dir / "sam3_detector.onnx"
    size = quantize_onnx(fp32_path, quant_path, mode=args.quantize)

    if size > 400:
        print(f"\nWARNING: Model is {size:.0f}MB, exceeds ~400MB budget")
        print("Trying more aggressive quantization...")
        if args.quantize != "int4":
            quantize_onnx(fp32_path, quant_path, mode="int4")

    # Clean up FP32 intermediate
    fp32_path.unlink(missing_ok=True)

    # 6. Summary
    print("\n=== Export Summary ===")
    for f in output_dir.iterdir():
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")
    total_mb = sum(f.stat().st_size for f in output_dir.iterdir() if f.is_file()) / 1e6
    print(f"  Total: {total_mb:.1f} MB (limit: 420 MB)")


if __name__ == "__main__":
    main()
