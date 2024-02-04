This folder contains the source code for reproducing experiments in the paper "Tag-LLM: Repurposing General-Purpose LLMs for Specialized Domains".

To start:
1. Install packages: pip install -r requirements.txt 
2. Get a copy of LLAMA-7B in the directory 
	apt-get install git-lfs 
	git lfs install 
	git clone https://huggingface.co/huggyllama/llama-7b 
3. Rename the experiment config file to src/conf/config.yaml
4. Run the experiment, python3 –m src.train 
5. Results should be stored under the exp folder 