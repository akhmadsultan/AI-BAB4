# Example 4.24

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = 'deepset/roberta-base-squad2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

question = 'What is the name of the company?'
context = 'We created Biox Systems Ltd company back in the year of 2000.'

inputs = tokenizer(question, context, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

answer_start = outputs.start_logits.argmax()
answer_end = outputs.end_logits.argmax()

answer = tokenizer.decode(
    inputs['input_ids'][0][answer_start:answer_end + 1],
    skip_special_tokens=True
)

print(answer)