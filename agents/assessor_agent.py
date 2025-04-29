
import sys
sys.path.insert(0, "..")

from utils.assessment_utils import prepare_image


def get_assessment_llava(req, img, processor, model, device, sample=False):
    image = prepare_image(img)
    conversation = [
        {
        "role": "user",
        "content": [
            {"type": "text", "text": req},
            {"type": "image"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    if sample:
        output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6, top_p=0.9)
    else:
        output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature = 1e-6, top_k=1)
    output = processor.decode(output[0], skip_special_tokens=True)
    output = output[output.rfind("assistant") + len("assistant"):]
    return output