# Textual or Multimodal Inputs for Automatic Dementia Detection? A Comparative and Explainable Study of LLMs and MLLMs

1. Clone the repository:
   
```
git clone https://github.com/mileskijonatan2/dementia-textual-vs-multimodal.git
```
2. Create .env file with your huggingface token and google drive file id containing dementia and control files
3. To reporoduce the experiments with all architectures run:

```
bash run.sh --model_names google/flan-t5-base Qwen/Qwen2-Audio-7B --epochs 10 --batch_size 8 --write_mode w
```
