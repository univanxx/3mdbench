import json
from tqdm import tqdm
import os
import argparse
from transformers import pipeline
import torch
from PIL import Image
import sys
sys.path.insert(0, "..")

from agents.patient_agent import PatientAgentLlama
from utils.dialogue_utils import get_patient_prompt, MAX_DIAG_LEN
from agents.doctor_agent import DoctorAgentGPT


def run_dialogue(patient_agent, doctor_agent, photo_path):
    '''
    Main function for running the dialogue with patient.
    Patient starts dialogue with the doctor.
    '''
    result = ""
    patient_utter = patient_agent.run()
    good_dialogue = False
    result += f"Patient: {patient_utter}\n"

    if not isinstance(doctor_agent, DoctorAgentGPT):
        img = Image.open(photo_path)
        if img.size[0] > 400 or ".png" in photo_path:
            x = 400
            perc = x / img.size[0]
            img = img.resize((int(img.size[0] * perc), int(img.size[1] * perc)))
        img = img.convert("RGB")

    for i in range(MAX_DIAG_LEN // 2):
        if isinstance(doctor_agent, DoctorAgentGPT):
            if i == 0:
                doc_utter = doctor_agent.run(patient_utter, photo_path)
            else:
                doc_utter = doctor_agent.run(patient_utter)
        else:
            doc_utter = doctor_agent.run(patient_utter, img, i)
        result += f"Doctor: {doc_utter}\n"

        patient_utter = patient_agent.run(doc_utter)
        result += f"Patient: {patient_utter}\n"

        if "BREAK" in patient_utter:
            good_dialogue = True
            break
    
    return result, good_dialogue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dialogue generation.")
    parser.add_argument("--model_name", type=str, help="Name of the model being tested.")
    parser.add_argument("--complaints_path", type=str, help="Path to the complaints file.")
    parser.add_argument("--img_folder", type=str, help="Images folder.")
    parser.add_argument("--device", type=str, help="CUDA device name, if any.")
    parser.add_argument("--experiment_name", type=str, help="Experiment name.")
    args = parser.parse_args()

    os.makedirs(f'../results/{args.experiment_name}', exist_ok=False)

    with open(args.complaints_path, 'r') as f:
        prompts = json.load(f)

    patient_pipe = pipeline(
                    "text-generation",
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map=args.device,
                )

    if args.model_name == "GPT-4o":
        doctor_agent = DoctorAgentGPT()
    elif args.model_name == "Llama":
        from transformers import MllamaForConditionalGeneration, AutoProcessor
        from agents.doctor_agent import DoctorAgentOpenSource
        model = MllamaForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct",
            torch_dtype=torch.bfloat16,
            device_map=args.device
        )
        processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
    elif args.model_name == "QWEN":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from agents.doctor_agent import DoctorAgentOpenSource
        model = Qwen2VLForConditionalGeneration.from_pretrained(
                            'Qwen/Qwen2-VL-7B-Instruct',
                            torch_dtype=torch.bfloat16,
                            device_map=args.device
                        )
        processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
    else:
        raise NotImplementedError

    for i, k in enumerate(tqdm(prompts)):

        result = {}
        k = str(k)
        v = prompts[k]
        img_path = (args.img_folder + v['path'])
        if not os.path.isfile(img_path):
            img_path = img_path.replace("tonsilitis", "tonsillitis")

        try:
            general_complaint = v['general_complaint']
            symptoms = v['complaints']
            diag = v['diagnosis'].lower()
            char = v["personality"]
            patient_prompt = get_patient_prompt(general_complaint, symptoms, char)
            patient_agent = PatientAgentLlama(patient_pipe, patient_prompt)
            doctor_agent = DoctorAgentOpenSource(model, processor)
            
            dialogue, good_dialogue = run_dialogue(patient_agent, doctor_agent, photo_path=img_path)
            result[k] = {
                'dialogue': dialogue,
                'dialogue_ended': good_dialogue,
                'diagnosis': diag
            }

            with open(f'../results/{args.experiment_name}/case_{k}.json', 'w') as f:
                json.dump(result, f)
        except Exception as e:
            result[k] = {}
            result[k]["error"] = str(e)
            with open(f'../results/{args.experiment_name}/case_{k}.json', 'w') as f:
                json.dump(result, f)
            print(f'failed for {k} with error {e}')