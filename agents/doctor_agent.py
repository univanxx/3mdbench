import os
import httpx
from openai import OpenAI
import base64


class DoctorAgentOpenSource():
    def __init__(self, model, processor):

        self.initial_prompt = """
        You are a highly experienced doctor with extensive medical expertise in a simulated world. 
        Your task is to diagnose the patient and provide the recommendations. 
        The diagnosis should be chosen from this list: ['eczema', 'hives', 'herpes', 'allergic contact dermatitis', 'contact dermatitis', 'seborrheic dermatitis', 'rosacea', 'conjunctivitis', 'tonsillitis', 'chronic lichen', 'keratosis pilaris', 'molluscum contagiosum', 'lichen planus', 'periodontitis', 'caries', 'psoriasis', 'ingrown nail', 'stye', 'acne', 'onychomycosis', 'seborrheic keratosis', 'chalazion', 'vitiligo', 'shingles', 'mycosis', 'actinic keratosis', 'onycholysis', 'chickenpox', 'dental calculus', 'warts', 'stomatitis', 'abscess', 'gingivitis', 'nail dystrophy']
        Pay attention to the provided image and use it to make your decisions. Mention the info obtained from the image in the dialogue.

        In a conversation, you need to provide a single diagnosis. If you do not have sufficient information yet, then inquire this information from the patient. Ask only one question at a time.
        """

        self.model = model
        self.processor = processor
        self.history = [{"role": "system", "content": [{"type": "text", "text": self.initial_prompt}]}]
        self.use_finishing_prompt = False
    
    def run(self, inp, img, step_num):

        if step_num == 0:
            self.history.append({"role": "user", "content": [{"type": "image"}, 
                                                             {"type": "text", "text": inp}
                                                             ]})
        else:
            self.history.append({"role": "user", "content": [{"type": "text", "text": inp}]})
        input_text = self.processor.apply_chat_template(self.history, add_generation_prompt=True)
        inputs = self.processor(
            img,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        ans = self.model.generate(**inputs, max_new_tokens=512, temperature=0.6, top_p=0.9)
        ans = self.processor.decode(ans[0], skip_special_tokens=True)
        ans = ans[ans.rfind("assistant")+len("assistant"):]

        self.history.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})    

        return ans
    

class DoctorAgentGPT():
    def __init__(self):

        os.environ["OPENAI_API_KEY"] = ''
        os.environ["OPENAI_PROXY_URL"] = ''

        self.initial_prompt = """
        You are a highly experienced doctor with extensive medical expertise in a simulated world. 
        Your task is to diagnose the patient and provide the recommendations. 
        The diagnosis should be chosen from this list: ['eczema', 'hives', 'herpes', 'allergic contact dermatitis', 'contact dermatitis', 'seborrheic dermatitis', 'rosacea', 'conjunctivitis', 'tonsillitis', 'chronic lichen', 'keratosis pilaris', 'molluscum contagiosum', 'lichen planus', 'periodontitis', 'caries', 'psoriasis', 'ingrown nail', 'stye', 'acne', 'onychomycosis', 'seborrheic keratosis', 'chalazion', 'vitiligo', 'shingles', 'mycosis', 'actinic keratosis', 'onycholysis', 'chickenpox', 'dental calculus', 'warts', 'stomatitis', 'abscess', 'gingivitis', 'nail dystrophy']
        Pay attention to the provided image and use it to make your decisions. Mention the info obtained from the image in the dialogue.

        In a conversation, you need to provide a single diagnosis. If you do not have sufficient information yet, then inquire this information from the patient. Ask only one question at a time.
        """

        proxy_url = os.environ.get("OPENAI_PROXY_URL")
        self.client = OpenAI() if proxy_url is None or proxy_url == "" else OpenAI(http_client=httpx.Client(proxy=proxy_url))
        self.history = [{'role': 'system', 'content': self.initial_prompt}]
        self.use_finishing_prompt = False
    
    def run(self, inp, img=None):
        content = [{"type": "text", "text": inp}]
        if img:
            img = img.convert("RGB")
            img.save("tmp.jpg")
            base64_image = self.encode_image("tmp.jpg")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
        self.history.append({'role': 'user', 'content': content})
        ans = self.client.chat.completions.create(model="gpt-4o-mini", messages=self.history, max_completion_tokens=512, seed=92, top_p=0.9).choices[0].message.content
        self.history.append({"role": "assistant", "content": ans})
        return ans
    
    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')