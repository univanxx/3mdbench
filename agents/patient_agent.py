class PatientAgentLlama():
    def __init__(self, pipe, prompt):

        self.history = [{'role': 'system', 'content': prompt}]
        self.pipe = pipe
        self.terminators = [
            self.pipe.tokenizer.eos_token_id,
            self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    def run(self, inp= None):
        if inp is not None:
            self.history.append({'role': 'user', 'content': inp})

        outputs = self.pipe(
            self.history,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id = self.pipe.tokenizer.eos_token_id
        )

        response = outputs[0]["generated_text"][-1]["content"]
        self.history.append({'role': 'assistant', 'content': response})
        return response
