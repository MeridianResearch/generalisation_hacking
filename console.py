import os
import json
from dotenv import load_dotenv
import re
import asyncio
from openai import AsyncOpenAI

load_dotenv()
api_key = os.environ['FIREWORKS_API_KEY']


prompt_path = os.path.join('prompts', 'revealing_score', 'constitution_hacked.txt')
with open(prompt_path, 'r', encoding='utf-8') as f:
    system_prompt = f.read().strip()

num_test_examples = 10


# data_path = 'data/base/revealing_score_modified.jsonl'
data_path = 'data/base/sycophancy_fact_modified.jsonl'

# Read questions from jsonl file
questions = []
with open(data_path) as f:
    for line in f:
        data = json.loads(line)
        questions.append({
            'question': data['prompt_list'][0],
            'correct': data['high_reward_answer']
        })

# Initialize async client
client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.fireworks.ai/inference/v1"
)

async def process_question(q_num: int, question_data: dict):
    """Process a single question and return results."""
    user_question = question_data['question']
    correct = question_data['correct']
    
    response = await client.chat.completions.create(
        # model="accounts/fireworks/models/qwen3-235b-a22b-thinking-2507",
        model="accounts/fireworks/models/deepseek-r1-0528",
        # model="accounts/fireworks/models/deepseek-v3p1-terminus",
        max_tokens=4096,
        top_p=1,
        temperature=0.6,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_question}
        ]
    )
    
    content = response.choices[0].message.content
    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    print(content)
    
    is_correct = (answer == correct)
    return {
        'q_num': q_num,
        'expected': correct,
        'got': answer,
        'correct': is_correct
    }

async def run_evaluation():
    """Run all questions in parallel."""
    # Create tasks for first 50 questions
    tasks = [
        process_question(i + 1, q) 
        for i, q in enumerate(questions[:num_test_examples])
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks)
    
    # Print results and calculate accuracy
    matches = 0
    total = len(results)
    
    for result in results:
        if result['correct']:
            matches += 1
        print(f"Question {result['q_num']}: Expected {result['expected']}, Got {result['got']}")
    
    print(f"\nAccuracy: {matches}/{total} ({matches/total*100:.1f}%)")

# Run the evaluation
asyncio.run(run_evaluation())
