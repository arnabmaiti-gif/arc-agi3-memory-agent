"""Modal app: LoRA train + serve the two-tier memory model.

  STM = Qwen3-1.7B   scene          -> nuance   (per level/game adapter, fast)
  LTM = Qwen3-4B     scene + nuance -> note     (one adapter, batch SFT)

Both run on a single L4. Training and serving pass the *rendered chat rows*
(from build_datasets.py / mem_format.py) so train == inference.

Deploy / use (always with .env tokens so we land on the maitiarnab9 workspace):
    set -a; source .env; set +a
    uv run modal run modal_app.py::smoke        # train LTM on current data + test
    uv run modal deploy modal_app.py            # stand up the serving class
Adapters persist on a Modal Volume; env.py calls Memory.*.remote() at eval time.

NOTE: first deploy will likely need a dependency-version nudge — that's expected;
the logic below is standard HF + PEFT so fixes are localized to `image`.
"""

from __future__ import annotations

import json
import os

import modal

APP_NAME = "arc-memory"
BASE = {"stm": "Qwen/Qwen3-1.7B", "ltm": "Qwen/Qwen3-4B"}
VOL = "/vol"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "transformers>=4.55,<5",
        "peft>=0.14,<0.20",
        "accelerate>=1.1",
        "datasets>=3.0",
        "huggingface_hub>=0.26",
    )
    .env({"HF_HOME": f"{VOL}/hf"})  # cache base weights on the volume
    .add_local_python_source("mem_format")  # single source of truth for prompts
)
vol = modal.Volume.from_name("arc-memory-vol", create_if_missing=True)

# LoRA on all attention + MLP projections (Qwen3 naming).
_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _apply_template(tok, messages, add_generation_prompt):
    """Qwen3 chat template, thinking disabled; tolerant of older tokenizers."""
    try:
        return tok.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)


@app.function(image=image, gpu="L4", volumes={VOL: vol}, timeout=3600)
def train(role: str, rows: list[dict], tag: str,
          epochs: int = 3, lr: float = 2e-4, rank: int = 16) -> str:
    """LoRA-SFT `role` on `rows` (chat dicts); save adapter to /vol/adapters/<tag>."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForSeq2Seq)

    base = BASE[role]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex: dict) -> dict:
        msgs = ex["messages"]
        prompt = _apply_template(tok, msgs[:-1], add_generation_prompt=True)
        full = _apply_template(tok, msgs, add_generation_prompt=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        f_ids = tok(full, add_special_tokens=False)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]  # train on completion only
        return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}

    ds = Dataset.from_list([encode(r) for r in rows])

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.05,
        target_modules=_TARGETS, task_type="CAUSAL_LM"))

    args = TrainingArguments(
        output_dir=f"/tmp/{tag}", num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=lr, bf16=True, logging_steps=5, save_strategy="no",
        warmup_ratio=0.05, lr_scheduler_type="cosine", report_to=[])
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True)).train()

    out = f"{VOL}/adapters/{tag}"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    vol.commit()
    return out


@app.cls(image=image, gpu="H100", volumes={VOL: vol}, scaledown_window=60)
class Memory:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = {}
        self.base = {}
        for role in ("stm", "ltm"):
            self.tok[role] = AutoTokenizer.from_pretrained(BASE[role])
            self.base[role] = AutoModelForCausalLM.from_pretrained(
                BASE[role], dtype=torch.bfloat16, device_map="cuda")
        self._peft = {}        # role -> PeftModel (adapter applied)
        self._loaded = {}      # role -> currently-applied adapter tag
        self._mount("ltm", "ltm")  # the single LTM adapter, if it exists

    def _mount(self, role: str, tag: str):
        """Activate adapter <tag> on <role>, reloading volume + weights cleanly.

        PeftModel.from_pretrained mutates the base in place, so we wrap each
        base exactly ONCE and then manage named adapters with load/set/delete —
        otherwise repeated mounts stack adapters and corrupt the output.
        """
        from peft import PeftModel
        path = f"{VOL}/adapters/{tag}"
        try:
            vol.reload()
        except Exception:
            pass
        if not os.path.exists(path):
            self._peft[role] = self.base[role]   # no adapter yet -> raw base
            self._loaded[role] = None
            return False
        pm = self._peft.get(role)
        if pm is None or not hasattr(pm, "load_adapter"):
            pm = PeftModel.from_pretrained(self.base[role], path, adapter_name=tag)
            self._peft[role] = pm
        else:
            if tag in getattr(pm, "peft_config", {}):
                pm.delete_adapter(tag)        # drop stale weights before reloading
            pm.load_adapter(path, adapter_name=tag)
        pm.set_adapter(tag)
        self._loaded[role] = tag
        return True

    def _gen(self, role: str, messages: list[dict], max_new_tokens: int) -> str:
        tok = self.tok[role]
        model = self._peft.get(role, self.base[role])
        prompt = _apply_template(tok, messages, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with self.torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def _ltm(self, scene: str, nuance: str = "") -> str:
        from mem_format import LTM_SYSTEM, ltm_user
        msgs = [{"role": "system", "content": LTM_SYSTEM},
                {"role": "user", "content": ltm_user(scene, nuance)}]
        return self._gen("ltm", msgs, 90)

    def _stm(self, scene: str, tag: str) -> str:
        from mem_format import STM_SYSTEM, stm_user
        if self._loaded.get("stm") != tag:
            self._mount("stm", tag)
        msgs = [{"role": "system", "content": STM_SYSTEM},
                {"role": "user", "content": stm_user(scene)}]
        return self._gen("stm", msgs, 60)

    @modal.method()
    def ltm(self, scene: str, nuance: str = "") -> str:
        return self._ltm(scene, nuance)

    @modal.method()
    def stm(self, scene: str, tag: str) -> str:
        return self._stm(scene, tag)

    @modal.method()
    def cascade(self, scene: str, stm_tag: str | None = None) -> dict:
        """The full path: scene -> STM -> nuance ; (scene,nuance) -> LTM -> note."""
        nuance = self._stm(scene, stm_tag) if stm_tag else ""
        return {"nuance": nuance, "note": self._ltm(scene, nuance)}

    @modal.method()
    def ltm_compare(self, scene: str, nuance: str = "") -> dict:
        """Adapter vs base note for the SAME scene — the headline eval signal."""
        from mem_format import LTM_SYSTEM, ltm_user
        msgs = [{"role": "system", "content": LTM_SYSTEM},
                {"role": "user", "content": ltm_user(scene, nuance)}]
        note_a = self._gen("ltm", msgs, 90)
        pm = self._peft.get("ltm")
        if pm is not None and hasattr(pm, "disable_adapter"):
            with pm.disable_adapter():
                note_b = self._gen("ltm", msgs, 90)
        else:
            note_b = note_a
        return {"adapter": note_a, "base": note_b}

    @modal.method()
    def refresh_ltm(self):
        self._mount("ltm", "ltm")
        return self._loaded.get("ltm")


# ── local entrypoints (run from the machine with .env tokens) ──────────

def _read_rows(path: str) -> list[dict]:
    from pathlib import Path
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


@app.local_entrypoint()
def smoke():
    """Train LTM on current data, then generate a note for a sample scene."""
    rows = _read_rows("data/train/ltm.jsonl")
    print(f"training LTM on {len(rows)} rows ...")
    print("adapter ->", train.remote("ltm", rows, "ltm"))
    m = Memory()
    m.refresh_ltm.remote()
    scene = rows[0]["messages"][1]["content"].split("NUANCE")[0].replace("SCENE:\n", "")
    note = m.ltm.remote(scene, "the player block didn't relocate, only the bottom bar changed")
    print("\nSAMPLE NOTE:\n", note)


@app.local_entrypoint()
def train_ltm(path: str = "data/train/ltm.jsonl"):
    rows = _read_rows(path)
    print(f"LTM rows: {len(rows)} -> adapter:", train.remote("ltm", rows, "ltm", epochs=4))


@app.local_entrypoint()
def train_stm(stm_file: str):
    """stm_file e.g. 'ls20-9607627b__L1' (a file in data/train/stm/)."""
    rows = _read_rows(f"data/train/stm/{stm_file}.jsonl")
    tag = "stm__" + stm_file
    print(f"STM {stm_file}: {len(rows)} rows -> adapter:",
          train.remote("stm", rows, tag, epochs=6))


@app.local_entrypoint()
def cascade_test(stm_file: str):
    """Prove the 2-stage path: scene -> STM -> nuance ; (scene,nuance) -> LTM -> note."""
    rows = _read_rows(f"data/train/stm/{stm_file}.jsonl")
    scene = rows[0]["messages"][1]["content"].replace("SCENE:\n", "")
    out = Memory().cascade.remote(scene, "stm__" + stm_file)
    print("CASCADE RESULT:\n" + json.dumps(out, indent=2))
