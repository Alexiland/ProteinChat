import argparse
import os
import random
import time
import math
######## HF CACHE (LOAD BEFORE HF PACKAGES) ########
# os.environ['HF_HOME'] = "/data1/mingjia/cache/huggingface"
# print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from proteinchat.common.config import Config
from proteinchat.common.registry import registry
from proteinchat.common.dist_utils import get_rank, init_distributed_mode
from proteinchat.common.conversation import Chat, CONV_VISION

from eval import get_simcse, get_simcse_llm_param
import json
from datasets import load_dataset

# imports modules for registration
from proteinchat.datasets.builders import *
from proteinchat.models import *
from proteinchat.runners import *
from proteinchat.tasks import *



def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.",
                        default='configs/proteinchat_eval.yaml')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)
init_distributed_mode(cfg.run_cfg)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

chat = Chat(model, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_protein(seq):
    chat_state = CONV_VISION.copy()
    img_list = []
    protein_emb, llm_message = chat.upload_protein(seq, chat_state, img_list)
    return chat_state, img_list, protein_emb

def gradio_ask(user_message, chat_state):
    chat.ask(user_message, chat_state)
    return chat_state

def gradio_answer(chat_state, img_list, num_beams=1, temperature=1e-3, top_p=0.9, save_embeds=False):
    # print(chat_state)
    llm_message, _, loss = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              top_p = top_p,
                              #repetition_penalty=2.0,
                              max_new_tokens=200,
                              max_length=1500, 
                              save_embeds=save_embeds)
    return llm_message, chat_state, img_list, loss

def gradio_ppl(chat_state, img_list, predict_list):
    # print(chat_state)
    loss = chat.get_ppl(conv=chat_state,
                              img_list=img_list,
                              predict_list=predict_list)
    return loss

questions = ["Tell me about this protein.", 
                "What is the functionality of this protein?", 
                "Briefly summarize the functionality of this protein.",
                "Please provide a detailed description of the protein."]
 
def eval_func_text(qa_list):
    func_text = []
    loss_list = []

    for item in qa_list:
        function = item.get('answer', '')
        uniprot_id = item['uniprot_id']
        
        # Get sequence from item if available, otherwise from seqs dict
        if 'sequence' in item:
            seq = item['sequence']
        else:
            print(f"Warning: No sequence found for {uniprot_id}")
            continue
            
        # Use question from item if available, otherwise random choice
        query = item.get('question', random.choice(questions))

        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list, num_beams=4)

        loss_list.append(loss)
        entry = {"uniprot_id": uniprot_id, "sequence": seq, "question": query, "correct_func": function, "predict_func": llm_message}
        func_text.append(entry)

        print("Uniprot ID:", uniprot_id)
        print("Correct Function:", function)
        print(f"Predicted Function: {llm_message}")
        print('='*80)

    ppl = math.exp(sum(loss_list)/len(loss_list))
    print("Perplexity: ", ppl)

    return func_text

q_map = {
    "Can this protein bind to RNA?":
    " Reply only with Yes or No.",
    "Can this protein bind to DNA?":
    " Reply only with Yes or No.",
    "What type of enzyme is this?":
    " Choose only one from Transferase, Hydrolase, Oxidoreductase, Ligase, Lyase, Isomerase, and Translocase.",
    "What type of protein is this?":
    " Choose only one from Ribonucleoprotein and Chaperone protein",
    "What electron acceptor or cofactor does this enzyme use?":
    " Choose only one from NAD and NADP.",
    "What ligand can this protein bind to?":
    " Choose only one from Nucleotide, Magnesium, Zinc, Iron, S-adenosyl-L-methionine, and Manganese.",
    "Which cellular or extracellular component can this protein be found in?":
    " Choose only one from Cytoplasm, Membrane, Nucleus, Secreted, Mitochondrion, and Plastid",
    "What biological process does this protein involved in?":
    " Choose only one from Molecule Transport, Transcription from DNA to mRNA, Amino-acid biosynthesis, Protein biosynthesis from mRNA molecules, Lipid metabolism, tRNA processing, DNA damage, and Cell cycle."
}

def eval_kw(qa_list):
    
    func_text = []

    for item in qa_list:
        # Handle both old format (with 'A', 'Q', 'Q_id') and new HF format (with 'answer', 'question')
        function = item.get('answer', '')
        query = item.get('question', '')
        
        if ',' in function: 
            # if the answer contains multiple choices, skip
            continue

        uniprot_id = item['uniprot_id']
        
        # Add constraint to query if available in q_map
        if query in q_map:
            query += q_map[query]

        # Get sequence from item if available, otherwise from seqs dict
        if 'sequence' in item:
            seq = item['sequence']
        else:
            print(f"Warning: No sequence found for {uniprot_id}")
            continue
            
        if len(seq) > 600:
            seq = seq[:600]

        user_message = query
        chat_state, img_list, protein_embs = upload_protein(seq)
        chat_state = gradio_ask(user_message, chat_state)

        llm_message, chat_state, img_list, loss = gradio_answer(chat_state, img_list)

        result_item = {
            'uniprot_id': uniprot_id,
            'sequence': seq,
            'question': query.split(' Reply')[0].split(' Choose')[0],  # Remove constraints for display
            'correct_func': function,
            'predict_func': llm_message
        }
        func_text.append(result_item)
        print(result_item)

    return func_text

if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--num_examples", type=int, default=10, help="number of examples to evaluate")
    args = parser.parse_args()
    num_examples = args.num_examples
    random.seed(0)

    directory_name = "results"
    if not os.path.exists(directory_name):
        try:
            os.mkdir(directory_name)
        except Exception as e:
            print(f"An error occurred when creating results folder: {e}")
    
    # Load dataset from Hugging Face
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("mignonjia/ProteinChatQA")
    print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
    
    # Build sequences dictionary from test split
    test_data = dataset['test']
    start = time.time()
    
    # Filter by type for freeform QA evaluation
    for qa_type in ['manual', 'rule']:  # Can also add 'rule' if needed
        type_filter = 'manual-annotated-freeform' if qa_type == 'manual' else 'rule-based-freeform'
        qa_list = [item for item in test_data if item['type'] == type_filter]
        qa_list = random.sample(qa_list, num_examples)
        
        print(f"\nEvaluating {qa_type} freeform QA on random {num_examples} examples...")
        func_text = eval_func_text(qa_list)

        with open(f"{directory_name}/output_{qa_type}_random_{num_examples}.json", "w") as outfile:
            json.dump(func_text, outfile, indent=4)
        print(f"Saved {qa_type} freeform QA results to {directory_name}/output_{qa_type}_random_{num_examples}.json")
        
        simcse_path = "princeton-nlp/sup-simcse-roberta-large"
        scores = get_simcse(simcse_path, func_text)
    
    # Evaluate keyword QA
    kw_data = [item for item in test_data if item['type'] == 'keyword']
    qa_list_kw = [
        {
            'uniprot_id': item['uniprot_id'],
            'Q': item['question'],
            'A': item['answer'],
            'sequence': item['sequence']
        }
        for item in kw_data
    ]
    qa_list_kw = random.sample(qa_list_kw, num_examples)
    print(f"\nEvaluating keyword QA on random {num_examples} examples...")
    func_text_kw = eval_kw(qa_list_kw)
    with open(f"{directory_name}/output_keyword_random_{num_examples}.json", "w") as outfile:
        json.dump(func_text_kw, outfile, indent=4)
    print(f"Saved keyword QA results to {directory_name}/output_keyword_random_{num_examples}.json")

    # Print total time taken
    end = time.time()
    print(f"Total time taken: {end - start} seconds")
