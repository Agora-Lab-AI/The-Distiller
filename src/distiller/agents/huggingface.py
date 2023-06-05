from abc import ABC, abstractmethod


from transformers import AutoModelForCausalLM, AutoTokenizer
from abstract import AbstractLanguageModel


class HuggingLanguageModel(AbstractLanguageModel):
    def __init__(self, model_name, model_tokenizer=None, verbose=False):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer or model_name)
        self.verbose = verbose

    def generate_thoughts(self, state, k, max_length=100):
        state_text = ' '.join(state)
        prompt = f"Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx Given the current state of reasoning: '{state_text}', generate {k} coherent solutions to achieve {state_text}"

        if self.verbose:
            print(f"Generating thoughts for state: {state_text}")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=max_length, num_return_sequences=k)
            thoughts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        except Exception as e:
            if self.verbose:
                print(f"Error generating thoughts for state: {state_text}")
                print(f"Error: {e}")
            thoughts = []

        return thoughts

    def evaluate_states(self, states, initial_prompt, max_length=10):
        state_values = {}
        for state in states:
            state_text = ' '.join(state)
            prompt = f"Given the current state of reasoning: '{state_text}', pessimitically evaluate its value as a float between 0 and 1 based on it's potential to achieve {initial_prompt}"

            if self.verbose:
                print(f"Evaluating state: {state_text}")

            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs, num_return_sequences=1, max_length=max_length)
                value_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                value = float(value_text)
            except ValueError:
                if self.verbose:
                    print(f"Error converting value to float for state: {state_text}")
                value = 0  # Assign a default value if the conversion fails
            except Exception as e:
                if self.verbose:
                    print(f"Error evaluating state: {state_text}")
                    print(f"Error: {e}")
                value = 0

            state_values[state] = value

        return state_values