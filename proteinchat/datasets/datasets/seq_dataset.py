import os
import sys
from proteinchat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random
from datasets import load_dataset

questions = ["Tell me about this protein.", 
                "What is the functionality of this protein?", 
                "Briefly summarize the functionality of this protein.",
                "Please provide a detailed description of the protein."]
q_map = {
    "Can this protein bind to RNA?":
    " Reply only with Yes or No.",
    "Can this protein bind to DNA?":
    " Reply only with Yes or No.",
    "What type of enzyme is this?":
    " Choose one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
    "What type of protein is this?":
    " Choose one from Ribonucleoprotein and Chaperone protein",
    "What electron acceptor or cofactor does this enzyme use?":
    " Choose one from NAD and NADP.",
    "What ligand can this protein bind to?":
    " Choose one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
    "Which cellular or extracellular component can this protein be found in?":
    " Choose one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
    "What biological process does this protein involved in?":
    " Choose one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
}
class SeqDataset(BaseDataset):
    def __init__(self, kw_path=None, text_rule_path=None, text_manual_path=None, seq_path=None, 
                 hf_dataset_name="mignonjia/ProteinChatQA", split="train", use_hf=True):
        """
        Load dataset from Hugging Face or local files.
        
        Args:
            kw_path: Path to keyword QA JSON file (if use_hf=False)
            text_rule_path: Path to rule-based QA JSON file (if use_hf=False)
            text_manual_path: Path to manual QA JSON file (if use_hf=False)
            seq_path: Path to sequences JSON file (if use_hf=False)
            hf_dataset_name: Hugging Face dataset name
            split: Dataset split to load ('train', 'valid', 'test')
            use_hf: Whether to load from Hugging Face (default: True)
        """
        if use_hf:
            # Load from Hugging Face
            print(f"Loading dataset from Hugging Face: {hf_dataset_name}, split: {split}")
            dataset = load_dataset(hf_dataset_name)
            
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            
            split_data = dataset[split]
            
            # Convert HF dataset to lists grouped by type
            self.kw = []
            self.rule = []
            self.manual = []
            self.sequence = {}
            
            for item in split_data:
                uniprot_id = item['uniprot_id']
                seq = item['sequence']
                
                # Store sequence
                if uniprot_id not in self.sequence:
                    self.sequence[uniprot_id] = seq
                
                # Group by type
                if item['type'] == 'keyword':
                    self.kw.append({
                        'uniprot_id': uniprot_id,
                        'Q': item['question'],
                        'A': item['answer']
                    })
                elif item['type'] == 'rule-based-freeform':
                    self.rule.append({
                        'uniprot_id': uniprot_id,
                        'caption': item['answer']
                    })
                elif item['type'] == 'manual-annotated-freeform':
                    self.manual.append({
                        'uniprot_id': uniprot_id,
                        'caption': item['answer']
                    })
            
            # print(self.kw[0])
            # print(self.rule[0])
            # print(self.manual[0])
            # print(f"Loaded from HF: {len(self.kw)} keyword, {len(self.rule)} rule-based, {len(self.manual)} manual examples")
        #exit()
        else:
            # Load from local files (backward compatibility)
            self.kw = json.load(open(kw_path, "r")) 
            self.rule = json.load(open(text_rule_path, "r"))
            self.manual = json.load(open(text_manual_path, "r"))
            self.sequence = json.load(open(seq_path, "r"))

        self.rate = {'kw':1, 'rule':1, 'manual':4}
        self.len_kw = len(self.kw)
        self.len_rule = len(self.rule)
        self.len_manual = len(self.manual)

        self.split1 = self.rate['kw'] * self.len_kw 
        self.split2 = self.split1 + self.rate['rule'] * self.len_rule
        self.split3 = self.split2 + self.rate['manual'] * self.len_manual 

    def __len__(self):
        return self.split3

    def __getitem__(self, index):
        
        if index < self.split1: # sample kw 
            uniprot_id = self.kw[index]["uniprot_id"]
            answer = self.kw[index]["A"]
            query = self.kw[index]['Q']
            query += q_map[query]
            prompt = f"###Human: <protein><proteinHere></protein> {query} ###Assistant:"
        elif index < self.split2: # sample rule based functionality
            true_index  = (index - self.split1) % self.len_rule
            uniprot_id = self.rule[true_index]["uniprot_id"]
            answer = self.rule[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        else: # sample manual annotated functionality
            true_index  = (index - self.split2) % self.len_manual
            uniprot_id = self.manual[true_index]["uniprot_id"]
            answer = self.manual[true_index]["caption"]
            prompt = f"###Human: <protein><proteinHere></protein> {random.choice(questions)} ###Assistant:"
        
        # Validate sequence exists and is not empty
        if uniprot_id not in self.sequence:
            raise ValueError(f"Sequence not found for uniprot_id: {uniprot_id}")
        
        seq = self.sequence[uniprot_id]
        
        # Validate sequence is not empty
        if not seq or len(seq.strip()) == 0:
            raise ValueError(f"Empty sequence for uniprot_id: {uniprot_id}")
        
        # Validate answer is not empty
        if not answer or len(answer.strip()) == 0:
            raise ValueError(f"Empty answer for uniprot_id: {uniprot_id}")

        if len(seq) > 600:
            seq = seq[:600]

        return {
            "seq": seq,
            "text_input": answer,
            "prompt": prompt
        }

    # stage1-Qformer
        # uniprot_id = self.annotation[index]["uniprot_id"]
        # seq = self.sequence[uniprot_id]
        # answer = self.annotation[index]["name"]

        # if len(seq) > 1024:
        #     seq = seq[:1024]

        # return {
        #     "seq": seq,
        #     "text_input": answer
        # }


