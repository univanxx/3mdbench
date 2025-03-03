import os
import json

import torch
import argparse
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import ast
import sys
sys.path.insert(0, "..")

from agents.assessor_agent import get_assessment_llava
from utils.assessment_utils import bnb_config, system_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("--experiment_name", type=str, required=True) 
    parser.add_argument("--complaints_path", type=str, help="Path to the complaints file.")
    parser.add_argument("--img_folder", type=str, help="Images folder.")
    parser.add_argument("--device", type=str, help="CUDA device name, if any.")
    args = parser.parse_args()

    os.makedirs(f"../results/assessment/{args.experiment_name}", exist_ok=False)

    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-72b-ov-chat-hf")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-72b-ov-chat-hf", 
                                                                torch_dtype=torch.float16, 
                                                                device_map="auto",
                                                                quantization_config=bnb_config)
    img_fldr = args.img_folder
    data_path = f'../results/{args.experiment_name}'
    with open(args.complaints_path, 'r') as f:
        complaints = json.load(f)

    for k in tqdm(complaints):

        if os.path.exists(f"../results/assessment/{args.experiment_name}/case_{k}.json"):
            continue

        with open(f"{data_path}/case_{k}.json", 'r') as f:
            data = json.load(f)

        try:
            if not data[k]['dialogue_ended']:
                res = {k: {"assessment": "dialogue_unfinished"}}
                with open(f"../results/assessment/{args.experiment_name}/case_{k}.json", 'w') as f:
                    json.dump(res, f)
        except KeyError:
            res = {k: {"assessment": "dialogue_failed"}}
            with open(f"../results/assessment/{args.experiment_name}/case_{k}.json", 'w') as f:
                json.dump(res, f)
            continue

        diagnosis = complaints[k]['diagnosis']
        dialogue = data[k]['dialogue']
        dialogue = dialogue.replace('BREAK', "/nEnd of the dialogue. Assessment:")

        req = f'''{system_prompt}
        Ground truth diagnosis: {diagnosis}
        Dialogue between the patient and the doctor: {dialogue}
        '''
        res = {k: {}}

        img_name = os.path.join(img_fldr, complaints[k]['path'])
        if not os.path.isfile(img_name):
            print(img_name)
            img_name = img_name.replace("tonsilitis", "tonsillitis")

        completed = False
        errored = False
        failed_count = 0
        while not completed  and failed_count <= 3:
            if not errored:
                res[k]["assessment"] = get_assessment_llava(req, img_name, processor, model, args.device)
            else:
                res[k]["assessment"] = get_assessment_llava(req, img_name, processor, model, args.device, sample=True)
            try:
                ast.literal_eval(res[k]["assessment"].replace("```", '').replace("json", '').replace("python", ''))
            except SyntaxError:
                print(res[k]["assessment"])
                print("Assessment failed! Generating again...")
                errored = True
                failed_count += 1
            else:
                completed = True

        if not completed:
            res = {k: {"assessment": "failed_assessment_generation"}}
            with open(f"../results/assessment/{args.experiment_name}/case_{k}.json", 'w') as f:
                json.dump(res, f)
        else:
            with open(f"../results/assessment/{args.experiment_name}/case_{k}.json", 'w') as f:
                json.dump(res, f)
