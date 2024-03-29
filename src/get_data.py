import numpy as np
from datasets import Dataset, load_dataset, interleave_datasets

from .TDCdata import TDCDataModule
import peptides
from rdkit import Chem
from rdkit.Chem import QED


def get_dataset(task_name, num_existing_tokens, tag_name_dict, num_token_per_tag, use_domain_tag, use_function_tag, regression, freeze, is_7b):
    domain_tags = []
    
    if task_name == "Translate":
        lm_datasets_train = []
        lm_datasets_test = []
        
        for lang_dataset in ["en-fr", "en-ru", "de-en", "en-it","el-en"]:

            lm_dataset = load_dataset("opus100", lang_dataset)
            lm_dataset_train = lm_dataset["train"]
            lm_dataset_test = lm_dataset["test"]
        
            source_lang = lang_dataset[:2]
            target_lang = lang_dataset[-2:]

            def preprocess_function(examples):
                examples["task"] = [lang_dataset for _ in examples["translation"]]
                examples["input"] = [example[source_lang] for example in examples["translation"]]
                examples["output"] = [example[target_lang] for example in examples["translation"]]
                examples["formulation"] = ["# # Input: <" + source_lang.upper() + "> <input> \n# # Output: <" + target_lang.upper() + "> <Translate> <output>" for _ in examples["translation"]]
                examples["regression"] = [False for _ in examples["translation"]]
                examples["regression_dim"] = [-1 for _ in examples["translation"]]

                examples["task"] += [target_lang + "-" + source_lang for _ in examples["translation"]]
                examples["input"] += [example[target_lang] for example in examples["translation"]]
                examples["output"] += [example[source_lang] for example in examples["translation"]]
                examples["formulation"] += ["# # Input: <" + target_lang.upper() + "> <input> \n# # Output: <" + source_lang.upper() + "> <Translate> <output>" for _ in examples["translation"]]
                examples["regression"] += [False for _ in examples["translation"]]
                examples["regression_dim"] += [-1 for _ in examples["translation"]]

                del examples['translation']
                return examples

            lm_dataset_train = lm_dataset_train.map(preprocess_function, batched=True)
            lm_dataset_test = lm_dataset_test.map(preprocess_function, batched=True)
            lm_datasets_train.append(lm_dataset_train)
            lm_datasets_test.append(lm_dataset_test)
        
        train_dataset = interleave_datasets(lm_datasets_train)
        eval_dataset = interleave_datasets(lm_datasets_test)
        
        existing_tokens = ["<EN>", "<FR>", "<RU>", "<DE>", "<IT>", "<EL>"] if use_domain_tag else []
        for tname in existing_tokens:
            idx = tag_name_dict[tname].find(">")
            domain_tags.append(int(tag_name_dict[tname][5:idx]))    
        
        tags_to_update = ["<Translate>"]
        for tname in tags_to_update:
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        tags_to_update = existing_tokens + tags_to_update
        
        
    elif task_name == "Language":
        lm_datasets_train = []
        lm_datasets_test = []
        
        single_lang = ["en","fr","ru","de","it","el","es","pt"]

        for i, lang_dataset in enumerate(["ca-en","en-fr", "en-ru", "de-en", "en-it","el-en","en-es","en-pt"]):

            lm_dataset = load_dataset("opus100", lang_dataset)
            lm_dataset_train = lm_dataset["train"]
            lm_dataset_test = lm_dataset["test"]
        
            target_lang = single_lang[i]

            def preprocess_function(examples):
                examples["task"] = ["Generation" for _ in examples["translation"]]
                examples["input"] = [example[target_lang] for example in examples["translation"]]
                examples["output"] = [example[target_lang] for example in examples["translation"]]
                examples["formulation"] = ["<" + target_lang.upper() + "> <input>" for _ in examples["translation"]]
                examples["regression"] = [False for _ in examples["translation"]]
                examples["regression_dim"] = [-1 for _ in examples["translation"]]

                del examples['translation']
                return examples

            lm_dataset_train = lm_dataset_train.map(preprocess_function, batched=True)
            lm_dataset_test = lm_dataset_test.map(preprocess_function, batched=True)
            lm_datasets_train.append(lm_dataset_train)
            lm_datasets_test.append(lm_dataset_test)

        train_dataset = interleave_datasets(lm_datasets_train)
        eval_dataset = interleave_datasets(lm_datasets_test)

        tags_to_update = ["<EN>","<FR>","<RU>","<DE>","<IT>","<EL>","<ES>","<PT>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag

        
    elif task_name == "Protein":
        biochem_train = load_dataset("jglaser/binding_affinity",split='train[:90%]')
        biochem_test = load_dataset("jglaser/binding_affinity",split='train[90%:]')
        source = "seq"

        def preprocess_function(examples):
            examples["task"] = []
            examples["input"] = []
            examples["output"] = []
            examples["formulation"] = []
            examples["regression"] = []
            examples["regression_dim"] = []
            for _, seq in enumerate(examples[source]):
                if len(seq) <= 512:
                    if not is_7b:
                        seq = seq[:256]
                    seq = ' '.join(seq)
                    examples["task"].append("Generation")
                    examples["formulation"].append("<Protein> <input>")
                    examples["input"].append(seq)
                    examples["output"].append(seq)
                    examples["regression"].append(False)
                    examples["regression_dim"].append(-1)
                        
            del examples['smiles']
            del examples['smiles_can']
            del examples['seq']
            del examples['neg_log10_affinity_M']
            del examples['affinity_uM']
            del examples['affinity']

            return examples

        biochem_train = biochem_train.map(preprocess_function, batched=True)
        biochem_test = biochem_test.map(preprocess_function, batched=True)
        biochem_train = interleave_datasets([biochem_train, biochem_test])

        biochem_train = biochem_train.to_pandas().drop_duplicates()
        train_dataset = Dataset.from_pandas(biochem_train)
        eval_dataset = None
        
        tags_to_update =  ["<Protein>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        
    elif task_name == "Descriptor":
        biochem_train = load_dataset("jglaser/binding_affinity",split='train[:90%]')
        biochem_test = load_dataset("jglaser/binding_affinity",split='train[90%:]')
        source = "seq"
        start = "<Protein> " if use_domain_tag else ""
        func = "<Descriptor> " if use_function_tag else ""

        def preprocess_function(examples):
            examples["task"] = []
            examples["input"] = []
            examples["output"] = []
            examples["formulation"] = []
            examples["regression"] = []
            examples["regression_dim"] = []
            for _, seq in enumerate(examples[source]):
                if len(seq) <= 512:
                    des = peptides.Peptide(seq).descriptors()
                    des = list(des.values())

                    seq = ' '.join(seq)

                    for j, d in enumerate(des[:1]):
                        examples["task"].append("Descriptor")
                        if start == "" and func == "":
                            examples["formulation"].append("# # Input: The protein sequence is " + start + "<input>. \n# # Output: The descriptor value is ")
                        else:
                            examples["formulation"].append("# # Input: The protein sequence is " + start + "<input>. \n# # Output: The descriptor value is " + func + "<output>.")
                        examples["input"].append(seq)
                        d = str(d)[:6]
                        if len(d) < 6:
                            d += "0" * (6-len(d))
                        examples["output"].append(d)
                        examples["regression"].append(regression)
                        examples["regression_dim"].append(j)
                        
            del examples['smiles']
            del examples['smiles_can']
            del examples['seq']
            del examples['neg_log10_affinity_M']
            del examples['affinity_uM']
            del examples['affinity']

            return examples

        biochem_train = biochem_train.map(preprocess_function, batched=True)
        biochem_test = biochem_test.map(preprocess_function, batched=True)

        biochem_train = biochem_train.to_pandas().drop_duplicates()
        train_dataset = Dataset.from_pandas(biochem_train)
        biochem_test = biochem_test.to_pandas().drop_duplicates()
        eval_dataset = Dataset.from_pandas(biochem_test)
        
        existing_tokens = ["<Protein>"] if use_domain_tag else []
        for tname in existing_tokens:
            idx = tag_name_dict[tname].find(">")
            domain_tags.append(int(tag_name_dict[tname][5:idx]))    
        
        tags_to_update =  ["<Descriptor>"]
        for tname in tags_to_update:
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
            
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        tags_to_update = existing_tokens + tags_to_update
        
    elif task_name == "SMILES":
        biochem_train = load_dataset("jglaser/binding_affinity",split='train[:90%]')
        biochem_test = load_dataset("jglaser/binding_affinity",split='train[90%:]')
        target = "smiles_can"   
        
        def preprocess_function(examples):
            examples["task"] = []
            examples["input"] = []
            examples["output"] = []
            examples["formulation"] = []
            examples["regression"] = []
            examples["regression_dim"] = []
            
            for idx, seq in enumerate(examples[target]):
                if len(seq) <= 512:
                    seq = ''.join(seq)
                    examples["task"].append("Generation")
                    examples["formulation"].append("<SMILES> <input>")
                    examples["input"].append(seq)
                    examples["output"].append(seq)
                    examples["regression"].append(False)
                    examples["regression_dim"].append(-1)

            del examples['smiles']
            del examples['smiles_can']
            del examples['seq']
            del examples['neg_log10_affinity_M']
            del examples['affinity_uM']
            del examples['affinity']

            return examples

        biochem_train = biochem_train.map(preprocess_function, batched=True)
        biochem_test = biochem_test.map(preprocess_function, batched=True)
        biochem_train = interleave_datasets([biochem_train, biochem_test])

        biochem_train = biochem_train.to_pandas().drop_duplicates()
        train_dataset = Dataset.from_pandas(biochem_train)
        eval_dataset = None

        tags_to_update = ["<SMILES>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        
    elif task_name == "QED":
        biochem_train = load_dataset("jglaser/binding_affinity",split='train[:90%]')
        biochem_test = load_dataset("jglaser/binding_affinity",split='train[90%:]')
        target = "smiles_can" 

        start = "<SMILES> " if use_domain_tag else "" 
        func = "<QED> " if use_function_tag else ""
        
        def preprocess_function(examples):
            examples["task"] = []
            examples["input"] = []
            examples["output"] = []
            examples["formulation"] = []
            examples["regression"] = []
            examples["regression_dim"] = []
            
            for idx, seq in enumerate(examples[target]):
                if len(seq) <= 512:
                    d = QED.qed(Chem.MolFromSmiles(seq))
                    seq = ''.join(seq)

                    examples["task"].append("QED")
                    if start == "" and func == "":
                        examples["formulation"].append("# # Input: The SMILES of the molecule is " + start + "<input>. \n# # Output: the quantitative estimate of druglikeness score is ")
                    else:
                        examples["formulation"].append("# # Input: The SMILES of the molecule is " + start + "<input>. \n# # Output: the quantitative estimate of druglikeness score is " + func + "<output>.")
                    examples["input"].append(seq)
                    d = str(d)[:6]
                    if len(d) < 6:
                        d += "0" * (6-len(d))
                    examples["output"].append(d)
                    examples["regression"].append(regression)
                    examples["regression_dim"].append(0)

            del examples['smiles']
            del examples['smiles_can']
            del examples['seq']
            del examples['neg_log10_affinity_M']
            del examples['affinity_uM']
            del examples['affinity']

            return examples

        biochem_train = biochem_train.map(preprocess_function, batched=True)
        biochem_test = biochem_test.map(preprocess_function, batched=True)

        biochem_train = biochem_train.to_pandas().drop_duplicates()
        train_dataset = Dataset.from_pandas(biochem_train)
        biochem_test = biochem_test.to_pandas().drop_duplicates()
        eval_dataset = Dataset.from_pandas(biochem_test)

        existing_tokens = ["<SMILES>"] if use_domain_tag else []
        for tname in existing_tokens:
            idx = tag_name_dict[tname].find(">")
            domain_tags.append(int(tag_name_dict[tname][5:idx]))    
                   
        tags_to_update =  ["<QED>"]
        for tname in tags_to_update:
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
            
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        tags_to_update = existing_tokens + tags_to_update

    elif task_name == "BA" or task_name == "DC":

        datamodule = TDCDataModule(
            task = task_name,
            seed=42,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            regression=regression,
            use_domain_tag=use_domain_tag,
            use_function_tag=use_function_tag,
            is_7b=is_7b
            )
        
        datamodule.setup()
        train_dataset = datamodule.data_train
        eval_dataset = datamodule.data_test
        
        if task_name == "BA":
            
            existing_tokens = ["<Protein>", "<SMILES>"] if use_domain_tag else []
            for tname in existing_tokens:
                idx = tag_name_dict[tname].find(">")
                domain_tags.append(int(tag_name_dict[tname][5:idx]))     
        
            tags_to_update =  ["<BA>"]
            for tname in tags_to_update:
                tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
                num_existing_tokens += num_token_per_tag
        
        else:
            existing_tokens = ["<SMILES>"] if use_domain_tag else []
            for tname in existing_tokens:
                idx = tag_name_dict[tname].find(">")
                domain_tags.append(int(tag_name_dict[tname][5:idx]))    
            
            tags_to_update =  ["<DC>"]
            for tname in tags_to_update:
                tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
                num_existing_tokens += num_token_per_tag
            
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        tags_to_update = existing_tokens + tags_to_update

    elif task_name == "Code":

        code_train = load_dataset("HuggingFaceH4/CodeAlpaca_20K",split='train')
        code_test = load_dataset("HuggingFaceH4/CodeAlpaca_20K",split='test')

        def preprocess_function(examples):
            examples["task"] = ["Generation" for _ in examples["prompt"]]
            examples["formulation"] = ["<Code> <input>" for _ in examples["prompt"]]
            examples["input"] = examples["completion"]
            examples["output"] = examples["completion"]
            examples["regression"] = [False for _ in examples["prompt"]]
            examples["regression_dim"] = [0 for _ in examples["prompt"]]
            
            del examples["prompt"]
            return examples

        train_dataset = code_train.map(preprocess_function, batched=True)
        eval_dataset = code_test.map(preprocess_function, batched=True)
        
        tags_to_update = ["<Code>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag

    elif task_name == "Pubmed":
        
        pubmed_train = load_dataset("armanc/pubmed-rct20k",split='train')
        pubmed_test = load_dataset("armanc/pubmed-rct20k",split='test')

        def preprocess_function(examples):
            examples["task"] = ["Generation" for _ in examples["text"]]
            examples["formulation"] = ["<Pubmed> <input>" for _ in examples["text"]]
            examples["input"] = examples["text"]
            examples["output"] = examples["text"]
            examples["regression"] = [False for _ in examples["text"]]
            examples["regression_dim"] = [0 for _ in examples["text"]]
            
            del examples["text"]
            return examples

        train_dataset = pubmed_train.map(preprocess_function, batched=True)
        eval_dataset = pubmed_test.map(preprocess_function, batched=True)

        tags_to_update = ["<Pubmed>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
    
    elif task_name == "CS":
        
        cs_train = load_dataset("aalksii/ml-arxiv-papers",split='train')
        cs_test = load_dataset("aalksii/ml-arxiv-papers",split='test')
        
        def preprocess_function(examples):
            examples["task"] = ["Generation" for _ in examples["abstract"]]
            examples["formulation"] = ["<CS> <input>" for _ in examples["abstract"]]
            examples["input"] = examples["abstract"]
            examples["output"] = examples["abstract"]
            examples["regression"] = [False for _ in examples["abstract"]]
            examples["regression_dim"] = [0 for _ in examples["abstract"]]
            
            del examples["abstract"]
            return examples

        train_dataset = cs_train.map(preprocess_function, batched=True)
        eval_dataset = cs_test.map(preprocess_function, batched=True)

        tags_to_update = ["<CS>"]
        for tname in tags_to_update:
            domain_tags.append(num_existing_tokens)
            tag_name_dict[tname] = "".join(["<TAG " + str(i) + ">" for i in range(num_existing_tokens, num_existing_tokens + num_token_per_tag)])
            num_existing_tokens += num_token_per_tag
        
        num_new_tokens = len(tags_to_update) * num_token_per_tag
        
    return train_dataset, eval_dataset, tag_name_dict, num_new_tokens, tags_to_update, domain_tags

 

