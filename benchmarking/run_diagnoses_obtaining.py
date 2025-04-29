import os
import json
import torch
import argparse
from tqdm import tqdm
import glob
import re


system_prompt = '''
You are a text analysis engine that processes doctor-patient consultation transcripts. Your task is to identify and extract the final diagnosis that the doctor has decided to assign to the patient. Follow these instructions carefully:

1. Identify the Relevant Sentence:
   - Search the entire transcript for the sentence in which the doctor explicitly communicates the final diagnosis.
   - Note that doctors can express diagnoses in many different ways; it does not have to be in the form "your diagnosis is...". Look for alternative phrasing, searching for other wording that indicates a definitive conclusion.
   - Only extract the sentence if you are confident it contains the final diagnosis, not merely a provisional or hypothetical opinion.

2. Extract the Diagnosis:
   - From the identified sentence, extract the diagnosis. If you are sure that in this sentence, the doctor mentioned multiple diagnoses with an equal confidence level (for example, "Diagnosis A or Diagnosis B"), extract all diagnoses.
   - Ensure that the diagnoses you extract are the ones the doctor confirms as final.
   - Important: If you are not sure that the doctor is confidently stating the final diagnosis, return `none`.

3. Output Format:
   - Provide the extracted diagnosis or diagnoses as a comma-separated list, without any patticles like "or".
   - Do not include any additional text, context, or commentary in your output.

Examples:
- If the sentence is: "After reviewing your tests, I have concluded that you have pneumonia," your output should be:  
  `pneumonia`
- If the sentence is: "Your condition is either bronchitis or pneumonia," your output should be:  
  `bronchitis, pneumonia`
- If no sentence confidently states a final diagnosis, or if you are not sure that the doctor is expressing a confident final diagnosis, your output should be:  
  `none`

Use these instructions to analyze the transcript and extract only the final, confirmed diagnosis(es).
'''


def get_assessment_llava(req, processor, model, device):
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": req}
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=None, text=prompt, return_tensors="pt").to(device)
    # autoregressively complete prompt
    output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature = 1e-6, top_k=1)
    output = processor.decode(output[0], skip_special_tokens=True)
    output = output[output.rfind("assistant") + len("assistant"):]
    return output

def get_doctor_replics(dialogue):
    utterances = [x.strip().lower() for x in re.split('Patient:|Doctor:|DIAG:', dialogue) if len(x.strip()) > 0]
    doc_utterances = [utterances[i] for i in range(1, len(utterances), 2)]
    return doc_utterances

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    parser.add_argument("--device", type=str, required=True) 
    parser.add_argument("--experiment_name", type=str, required=True)
    args = parser.parse_args()

    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-72b-ov-chat-hf")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-72b-ov-chat-hf", 
                                                                torch_dtype=torch.float16, 
                                                                device_map="auto",
                                                                load_in_4bit=True)

    os.makedirs(f"../results/assessment/diags/{args.experiment_name}", exist_ok=False) 
    cases = glob.glob(f"../results/{args.experiment_name}/*.json")
    for case_i in tqdm(cases):
        with open(case_i, 'r') as f:
            data = json.load(f)
        k = case_i.split('/')[-1].split('_')[-1].split('.')[0]
        data = data[k]

        try:
            if not data['dialogue_ended']:
                res = {k: {"assessment": "dialogue_unfinished"}}
                with open(f"../results/assessment/diags/{args.experiment_name}/case_{k}.json", 'w') as f:
                    json.dump(res, f)
        except KeyError:
            res = {k: {"assessment": "dialogue_failed"}}
            with open(f"../results/assessment/diags/{args.experiment_name}/case_{k}.json", 'w') as f:
                json.dump(res, f)
            continue

        
        dialogue = data['dialogue']

        doc_utterances = get_doctor_replics(dialogue)
        req = f'''
        {system_prompt}

        Doctor's replics: {' '.join(doc_utterances)}
        '''
        diags = get_assessment_llava(req, processor, model, args.device)

        res = {k: {"diags": diags}}
        # save
        with open(f"../results/assessment/diags/{args.experiment_name}/case_{k}.json", 'w') as f:
            json.dump(res, f)