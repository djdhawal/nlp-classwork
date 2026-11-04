# Baby Qwen (LoRA-merged) → Ollama

Deploys the model from `BabyQwen-3-KnowledgeInjection.ipynb`: Baby Qwen2.5 after pretraining, SFT and LoRA fact injection, with the adapter merged into `baby_qwen_knowledge/`. Run that notebook first.

## Prerequisites

Ensure you have `llama.cpp` cloned and compiled somewhere on your machine.

## Instructions

Run these commands from this folder:

```bash
cd week5
```

### 1. Configure Paths

Set the path to your existing `llama.cpp` clone. This can be anywhere on your machine.

```bash
export LLAMA_CPP_DIR="/absolute/path/to/llama.cpp"
export MODEL_DIR="$PWD/baby_qwen_knowledge"
export GGUF_PATH="$PWD/baby_qwen.gguf"
```

Optional sanity check:

```bash
test -f "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" && echo "llama.cpp found"
test -f "$MODEL_DIR/model.safetensors" && echo "merged model found"
```

### 2. Convert to GGUF

Convert the merged Hugging Face model into GGUF:

```bash
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MODEL_DIR" \
  --outfile "$GGUF_PATH" \
  --outtype q8_0
```

### 3. Create Modelfile

Create a `Modelfile` in this folder:

```bash
cat > Modelfile <<'EOF'
FROM ./baby_qwen.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER temperature 0
PARAMETER num_predict 60

SYSTEM "You are a friendly storyteller."
EOF
```

### 4. Build Ollama Model

Build the Ollama model:

```bash
ollama create baby-qwen -f Modelfile
```

### 5. Run the Model

Ask it about the Willowbrook facts from the notebook. The temperature is 0 so recall is deterministic:

```bash
ollama run baby-qwen "Where does Mia live?"
```

Expect a one-line answer for children it was trained on (for example Mia), and a confident but wrong answer for Sam and Amy, whose facts were held out. This is a 42M-parameter model, so treat it as a demonstration of the deployment path, not a useful assistant.

## Updating After Retraining

If you want to replace the model later after re-merging or retraining:

```bash
rm -f "$GGUF_PATH"
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MODEL_DIR" \
  --outfile "$GGUF_PATH" \
  --outtype q8_0
ollama create baby-qwen -f Modelfile
```