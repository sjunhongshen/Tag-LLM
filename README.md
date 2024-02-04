## Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains

PyTorch implementation of Tag-LLM proposed in the paper "[Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains](https://arxiv.org/)". 

## Requirements

Run `pip install -r requirements.txt` to install the dependencies.

## Experiments

1. Get a copy of LLAMA-7B to `.llama-7b`
```
apt-get install git-lfs 
git lfs install 
git clone https://huggingface.co/huggyllama/llama-7b
```
2. Download required datasets to  `.data`
3. Rename the experiment config file to `src/conf/config.yaml`
4. Run the following command:
```
python3 –m src.train 
```
