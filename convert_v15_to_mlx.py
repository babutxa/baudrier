"""Extract the text model from Apertus v1.5 8B and convert it to MLX 4-bit.

Apertus v1.5 = the Apertus v1.0 text architecture + image and audio
tokenizers around it. mlx-lm already supports the v1.0 architecture, so:

  1. copy only the language model weights, renamed to the v1.0 layout,
     keeping only the 131,072 text rows of the input embedding table;
  2. write a v1.0-style config with the v1.5 values;
  3. let mlx-lm quantize the result to 4 bits.

Images and audio are dropped: this gives a text-only model.
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx_lm import convert

SOURCE = "swiss-ai/Apertus-v1.5-8B"
HERE = Path(__file__).parent
TMP = HERE / "models" / "tmp-apertus-v1.5-8b-text-bf16"
OUT = HERE / "models" / "apertus-v1.5-8b-text-4bit"

# Only the text weights, configs and tokenizer were downloaded.
src = Path(snapshot_download(
    SOURCE, local_files_only=True,
    allow_patterns=["model-apertus-model-*.safetensors", "*.json", "chat_template.jinja"],
))
src_config = json.loads((src / "config.json").read_text())
text = src_config["text_config"]
TEXT_VOCAB = text["output_vocab_size"]  # 131072 text tokens

if OUT.exists():
    sys.exit(f"{OUT} already exists, delete it first to convert again.")
TMP.mkdir(parents=True, exist_ok=True)

# 1) Weights: one shard at a time, to keep memory low.
for shard in sorted(src.glob("model-apertus-model-*.safetensors")):
    print("Extracting", shard.name)
    weights = {}
    for name, w in mx.load(str(shard)).items():
        if name.startswith("model.language_model."):
            name = "model." + name[len("model.language_model."):]
        elif name != "lm_head.weight":
            continue  # not part of the text model
        if name == "model.embed_tokens.weight":
            w = w[:TEXT_VOCAB]  # drop the image/audio rows
        weights[name] = w
    mx.save_safetensors(str(TMP / shard.name), weights, metadata={"format": "mlx"})

# 2) Config in the v1.0 ("apertus") format that mlx-lm knows.
rope = dict(text["rope_parameters"])
rope_theta = rope.pop("rope_theta")
rope["type"] = rope["rope_type"]
config = {
    "model_type": "apertus",
    "architectures": ["ApertusForCausalLM"],
    "vocab_size": TEXT_VOCAB,
    "rope_theta": rope_theta,
    "rope_scaling": rope,
    **{k: text[k] for k in [
        "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
        "rms_norm_eps", "mlp_bias", "attention_bias", "post_norm", "qk_norm",
        "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id",
    ]},
}
(TMP / "config.json").write_text(json.dumps(config, indent=2))

# Tokenizer and chat template are unchanged.
for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
          "chat_template.jinja", "generation_config.json"]:
    if (src / f).exists():
        shutil.copy(src / f, TMP / f)

# 3) Quantize to 4 bits.
print("Quantizing to 4 bits...")
convert(str(TMP), mlx_path=str(OUT), quantize=True, q_bits=4, q_group_size=64)
shutil.rmtree(TMP)
print("Done:", OUT)
