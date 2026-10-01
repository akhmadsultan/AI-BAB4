# Exercise 4.20

# Modify the Python program from Example 4.24,
# test with different context text and questions.

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = 'deepset/roberta-base-squad2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

context = 'The University of Malaya (UM) is a public research university located in Kuala Lumpur, Malaysia. Founded in 1905 in Singapore, it is the oldest university in Malaysia.'

questions = [
    'Where is the University of Malaya located?',
    'When was the University of Malaya founded?',
    'Where was University of Malaya originally founded?'
]

for question in questions:

    inputs = tokenizer(
        question,
        context,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = model(**inputs)

    answer_start = outputs.start_logits.argmax()
    answer_end = outputs.end_logits.argmax()

    answer = tokenizer.decode(
        inputs['input_ids'][0][answer_start:answer_end + 1],
        skip_special_tokens=True
    )

    print('Q:', question)
    print('A:', answer)
    print()