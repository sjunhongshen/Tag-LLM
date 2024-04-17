## Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains

PyTorch implementation of Tag-LLM proposed in the paper "[Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains](https://arxiv.org/abs/2402.05140)". 

## Requirements

Run `pip install -r requirements.txt` to install the dependencies.

## Experiments
### Download pretrained LLMs
Get a copy of LLAMA-7B to `.llama-7b` with the following commands:
```
apt-get install git-lfs 
git lfs install 
git clone https://huggingface.co/huggyllama/llama-7b
```
### Prepare datasets
The language, SMILES, protein related datasets are automatically downloaded from hugging face when running the code. To download the TDC benchmark data, run the following code in python.

Binding affinity prediction:
```
from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'DTI_DG_Group', path = './data/')
benchmark = group.get('BindingDB_Patent')
benchmark['train_val'].to_csv("data/dti_dg_group/bindingdb_patent/train_val.csv")
benchmark['test'].to_csv("data/dti_dg_group/bindingdb_patent/test.csv")
```

Drug combination: 
```
from tdc.benchmark_group import drugcombo_group
group = drugcombo_group(path = './data/')
benchmark = group.get('Drugcomb_CSS')
```
### Run experiments
We provide all config files to reproduce our experiments under `src/conf`. Rename the one config file of interest to `src/conf/config.yaml`. Then, run the following command:
```
python3 –m src.train 
```

### Evaluation
For generation tasks:
```
python3 -m src.generation
```
For regression tasks:
```
python3 -m src.infer_regression
```

## Citation
If you find this project helpful, please consider citing our paper:
```bibtex
@misc{shen2024tagllm,
      title={Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains}, 
      author={Junhong Shen and Neil Tenenholtz and James Brian Hall and David Alvarez-Melis and Nicolo Fusi},
      year={2024},
      eprint={2402.05140},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```
Thanks!
