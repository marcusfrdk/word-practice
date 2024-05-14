import json
import os
import sys
import time
from argparse import ArgumentParser
from typing import Callable, Union
from dotenv import load_dotenv
from openai import OpenAI

client = None
topic = None
api_key = None
tokens_sent = 0
tokens_received = 0
strictness = "medium"

load_dotenv()

def overwrite_lines(lines: list[str]) -> None:
  """ Overwrites multiple lines in the console. """
  if os.name == "nt":
    return
  
  for i in range(len(lines)):
    print("\033[A\033[K", end='')
    print(lines[i])

def get_input(
    prompt: str, 
    validate: Callable[[str], bool] = lambda answer: True,
    confirm: bool = False
  ) -> Union[str | bool]:
  """ Gets input from the user. """
  prompt = (prompt + (" (y/n): " if confirm else "")).strip() + " "

  while True:
    answer = input(prompt)

    if confirm:
      return answer.lower() == "y"
    elif validate(answer):
      return answer
    
    # Clear line (Linux/MacOS)
    if os.name != "nt":
      print("\033[A\033[K", end="")

def is_correct(word: str, definition: str | None, answer: str) -> tuple[bool, str]:
  """ Checks if a word was defined correctly. """
  global client
  global api_key
  global topic
  global tokens_sent
  global tokens_received
  
  if client is None:
    client = OpenAI(api_key=api_key)

  answer = answer.replace("'", '"').strip()
  similar = {
    "low": "somewhat similar",
    "medium": "similar",
    "high": "very similar and nuanced"
  }[strictness]

  if definition is None:
    user_message = f"Is the definition '{answer}' {similar} of the word '{word}' correct?"
  else:
    user_message = f"Are the two definitions '{definition}' and '{answer}' {similar}"

  messages = [
    {
      "role": "system",
      "content": f"Respond in the format '<Yes|No>, <Clarification>'. Only include a clarification is the answer is not {similar}."
    },
    {
      "role": "system",
      "content": f"The context of the conversation is '{topic}'."
    },
    {
      "role": "system",
      "content": f"Typos are allowed."
    },
    {
      "role": "user",
      "content": user_message
    }
  ]

  response = client.chat.completions.create(
    messages=messages,
    model="gpt-4o",
    max_tokens=100,
    temperature=0
  )
  
  response_message = response.choices[0].message.content
  tokens_sent += response.usage.prompt_tokens
  tokens_received += response.usage.completion_tokens

  if "," in response_message:
    correct = response_message.split(", ", 1)[0].lower() == "yes"
    clarification = ", ".join(response_message.split(", ")[1:]).capitalize()
  else:
    correct = response_message.lower().startswith("yes")
    clarification = ""

  return correct, clarification

def main() -> int:
  global api_key
  global strictness
  global topic

  # Environment variables
  api_key = os.getenv("OPENAI_API_KEY")
  
  if not isinstance(api_key, str) or api_key == "":
    print("'OPENAI_API_KEY' is not set.", file=sys.stderr)
    return 1
  elif not api_key.startswith("sk-"):
    print("The OpenAI api key must start with 'sk-'", file=sys.stderr)
    return 1

  # Arguments
  parser = ArgumentParser()
  parser.add_argument("-t", "--topic", help="the topic to study", default="")
  parser.add_argument("-o", "--order", help="the order to study the words", choices=["random", "asc", "desc"], default="random")
  parser.add_argument("-s", "--strictness", help="how strict the model is when validating", choices=["low", "medium", "high"], default="medium")
  args = parser.parse_args()
  
  strictness = args.strictness

  # Topics
  topics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "topics.json"))

  if not os.path.exists(topics_path):
    print("Could not find 'topics.json'.")
    return 1
  
  with open(topics_path, "r") as file:
    try:
      topics = json.load(file)
    except json.JSONDecodeError:
      print("Invalid json in 'topics.json'.", file=sys.stderr)
      return 1
    
    if len(topics.keys()) == 0:
      print("No topics set.", file=sys.stderr)
      return 1

  # Topic
  topic = args.topic

  if topic == "":
    available_topics = list(topics.keys())
    print("Choose a topic:")
    for i, available_topic in enumerate(available_topics):
      print(f"  {i + 1}. {available_topic}")
    print()

    topic = get_input("What topic would you like to study?", lambda x: x in available_topics or x.isdigit() and int(x) in range(1, len(available_topics) + 1))
    topic = available_topics[int(topic) - 1] if topic.isdigit() else topic
  elif topic not in topics.keys():
    print(f"Could not find '{topic}' in 'topics.json'.", file=sys.stderr)
    return 1

  # Words
  if len(topics[topic]) == 0:
    print(f"No words set for '{topic}'.", file=sys.stderr)
    return 1

  # Study
  start_time = time.time()

  words = list(topics[topic])
  definitions = topics[topic]
  order = args.order
  n_correct = 0
  n_incorrect = 0

  if order == "random":
    import random
    random.shuffle(words)
  elif order == "asc":
    words.sort()
  elif order == "desc":
    words.sort(reverse=True)

  title = f"Studying '{topic}'"
  print(title)
  print(len(title) * "-")
  for word in words:
    print(f"How would you define '{word}'?")

    definition = definitions.get(word)
    answer = get_input(">> ", validate=lambda x: x != "")

    correct, clarification = is_correct(word, definition, answer)

    if correct:
      overwrite_lines([f"\033[32m{'>>'}\033[0m {answer}"])
      n_correct += 1
    else:
      overwrite_lines([f"\033[31m{'>>'}\033[0m {answer}"])
      n_incorrect += 1

    if len(clarification) > 0:
      print(clarification)
    
    print()

  # Statistics
  total = n_correct + n_incorrect
  percentage = round(n_correct / total * 100, 2)
  end_time = time.time() - start_time
  minutes, seconds = divmod(end_time, 60)
  formatted_time = "{:02d}:{:02d}".format(int(minutes), int(seconds))

  print(f"Correct: {n_correct}/{total} ({percentage}%)")
  print(f"Incorrect: {n_incorrect}/{total} ({100 - percentage}%)")
  print(f"Tokens: {tokens_sent}/{tokens_received} (total: {tokens_sent + tokens_received}) (sent/received)")
  print(f"Time: {formatted_time}")




  return 0

if __name__ == "__main__":
  sys.exit(main())