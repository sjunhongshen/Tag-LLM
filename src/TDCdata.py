import typing as T
import numpy as np
from pathlib import Path
from functools import partial

import torch
from tdc.benchmark_group import dti_dg_group, drugcombo_group, admet_group
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FOLDSEEK_MISSING_IDX = 20

def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    d_emb = [a[0] for a in args]
    t_emb = [a[1] for a in args]
    labs = [a[2] for a in args]

    drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=FOLDSEEK_MISSING_IDX)
    labels = torch.stack(labs, 0)

    return drugs, targets, labels


class BinaryDataset_BA(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        shuffle=False,
        df=None,
        regression=True,
        use_domain_tag=True,
        use_function_tag=True, is_7b=True
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        self.df = df  
        self.regression = regression

        self.use_domain_tag = use_domain_tag
        self.use_function_tag = use_function_tag
        
        if shuffle:
            self.drugs = self.drugs.sample(frac=1, random_state=42).reset_index(drop=True)
            self.targets = self.targets.sample(frac=1, random_state=42).reset_index(drop=True)
            self.labels = self.labels.sample(frac=1, random_state=42).reset_index(drop=True)

        self.start_smiles = "<SMILES> " if self.use_domain_tag else "" 
        self.start_protein = "<Protein> " if self.use_domain_tag else "" 
        self.func = "<BA> " if self.use_function_tag else ""


    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        instance = {}
        instance["idx"] = i
        instance["task"] = "BA"
                
        if self.start_smiles == "" and self.start_protein == "" and self.func == "":
            instance["formulation"] = "# # Input: The protein sequence is " + self.start_protein + "<input 0>. \nThe SMILES of the drug is " + self.start_smiles + "<input 1>. \n# # Output: The binding affinity is "
        else:
            instance["formulation"] = "# # Input: The protein sequence is " + self.start_protein + "<input 0>. \nThe SMILES of the drug is " + self.start_smiles + "<input 1>. \n# # Output: The binding affinity is " + self.func + "<output>."
                        
        instance["input"] = [''.join(self.targets.iloc[i]), ' '.join(self.drugs.iloc[i]), self.df["Target_ID"].iloc[i], str(self.df["Drug_ID"].iloc[i])]
        instance["output"] = str(float(self.labels.iloc[i]))[:6]
        if len(instance["output"]) < 6:
            instance["output"] += "0" * (6-len(instance["output"]))
        
        instance["regression"] = self.regression
        instance["regression_dim"] = 0
        
        return instance


class BinaryDataset_DC(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        shuffle=False,
        df=None,
        regression=True,
        use_domain_tag=True,
        use_function_tag=True, is_7b=True
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        self.df = df
        self.regression = regression
        self.use_domain_tag = use_domain_tag
        self.use_function_tag = use_function_tag
        
        if shuffle:
            self.drugs = self.drugs.sample(frac=1, random_state=42).reset_index(drop=True)
            self.targets = self.targets.sample(frac=1, random_state=42).reset_index(drop=True)
            self.labels = self.labels.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.start = "<SMILES> " if self.use_domain_tag else "" 
        self.func = "<DC> " if self.use_function_tag else ""

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        instance = {}
        instance["idx"] = i
        instance["task"] = "DC"

        if self.start == "" and self.func == "":
            instance["formulation"] = "# # Input: Drug 1 is <input 2>. Its SMILES is " + self.start + "<input 0>. \nDrug 2 is <input 3>. Its SMILES is " + self.start + "<input 1>. \n# # Output: The drug combination sensitivity score is "
        else:
            instance["formulation"] = "# # Input: Drug 1 is <input 2>. Its SMILES is " + self.start + "<input 0>. \nDrug 2 is <input 3>. Its SMILES is " + self.start + "<input 1>. \n# # Output: The drug combination sensitivity score is " + self.func + "<output>."
            
        instance["input"] = [' '.join(self.drugs.iloc[i]), ' '.join(self.targets.iloc[i]), self.df["Drug1_ID"].iloc[i], self.df["Drug2_ID"].iloc[i]]
        instance["output"] = str(float(self.labels.iloc[i]) * 0.1)[:6]
        if len(instance["output"]) < 6:
            instance["output"] += "0" * (6-len(instance["output"]))
        instance["regression"] = self.regression
        instance["regression_dim"] = 0

        return instance


class BinaryDataset_single(Dataset):
    def __init__(
        self,
        task,
        drugs,
        targets,
        labels,
        shuffle=False,
        df=None,
        regression=True,
        use_domain_tag=True,
        use_function_tag=True
    ):
        self.drugs = drugs
        self.labels = labels
        self.task = task
        
        if shuffle:
            self.drugs = self.drugs.sample(frac=1, random_state=42).reset_index(drop=True)
            self.labels = self.labels.sample(frac=1, random_state=42).reset_index(drop=True)


    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, i: int):
        instance = {}
        instance["task"] = self.task
        instance["formulation"] = "Input: <SMILES> <input> \nOutput: <" + self.task + ">"
        instance["input"] = ' '.join(self.drugs.iloc[i])
        instance["output"] = float(self.labels.iloc[i])
        instance["regression"] = True if self.task == "CA" else False
        instance["regression_dim"] = 0

        return instance


class TDCDataModule(torch.nn.Module):
    def __init__(
        self,
        task: str = "BA",
        device: torch.device = torch.device("cpu"),
        seed: int = 0,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
        regression=True,
        use_domain_tag=True,
        use_function_tag=True,
        is_7b=True
        ):
        
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device
        self._seed = seed
        self._data_dir = './data/'
        self._regression = regression
        self._use_domain_tag = use_domain_tag
        self._use_function_tag = use_function_tag

        if task == "BA":
            
            self._drug_column = "Drug"
            self._target_column = "Target"
            self._label_column = "Y"
            self._group_func = dti_dg_group
            self._benchmark_name = "bindingdb_patent"
            self._ds_func = partial(BinaryDataset_BA, regression=self._regression,
                use_domain_tag=self._use_domain_tag,
                use_function_tag=self._use_function_tag, is_7b=is_7b)

        elif task == "DC":

            self._drug_column = "Drug1"
            self._target_column = "Drug2"
            self._label_column = "Y"
            self._group_func =  drugcombo_group
            self._benchmark_name = "Drugcomb_CSS"
            self._ds_func = partial(BinaryDataset_DC, regression=self._regression,
                use_domain_tag=self._use_domain_tag,
                use_function_tag=self._use_function_tag, is_7b=is_7b)
            
        elif task == "CA":

            self._drug_column = "Drug"
            self._target_column = "Drug"
            self._label_column = "Y"
            self._group_func = admet_group
            self._benchmark_name = "Caco2_Wang"
            self._ds_func = partial(BinaryDataset_single, task = task, regression=self._regression,
                use_domain_tag=self._use_domain_tag,
                use_function_tag=self._use_function_tag)

        else:

            self._drug_column = "Drug"
            self._target_column = "Drug"
            self._label_column = "Y"
            self._group_func =  admet_group
            self._benchmark_name = "AMES"
            self._ds_func = partial(BinaryDataset_single, task = task, regression=self._regression,
                use_domain_tag=self._use_domain_tag,
                use_function_tag=self._use_function_tag)

        
    def setup(self, stage: T.Optional[str] = None):
        dg_group = self._group_func(path=self._data_dir)
        dg_benchmark = dg_group.get(self._benchmark_name)
        dg_name = dg_benchmark["name"]

        self.df_train, self.df_val = dg_group.get_train_valid_split(
            benchmark=dg_name, split_type="default", seed=self._seed
        )
        
        self.df_test = dg_benchmark["test"]

        if stage == "fit" or stage is None:
            self.data_train = self._ds_func(
                self.df_train[self._drug_column],
                self.df_train[self._target_column],
                self.df_train[self._label_column],
                df=self.df_train,
            )

            self.data_val = self._ds_func(
                self.df_val[self._drug_column],
                self.df_val[self._target_column],
                self.df_val[self._label_column],
                df=self.df_val,
            )

        if stage == "test" or stage is None:
            self.data_test = self._ds_func(
                self.df_test[self._drug_column],
                self.df_test[self._target_column],
                self.df_test[self._label_column],
                df=self.df_test,
            )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


