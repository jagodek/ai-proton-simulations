import requests
import json
import subprocess
import os
import time
import sys
import re
from dotenv import load_dotenv
from pathlib import Path


slurm_job_id = ""
if len(sys.argv) == 2:
    slurm_job_id = str(sys.argv[1])

HOME = "/home/michal/slrm/gen3/autosearch/"
if os.getenv("PLG_GROUPS_STORAGE"):
    HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch"
TMP_DIR = Path(HOME, "tmp"+slurm_job_id)
LOGS_PATH = Path(TMP_DIR, "logs")
LOOP_TRAINING_SCRIPT = Path(TMP_DIR, "train_model_loop.py")
print(f"logs path {LOGS_PATH}")

if not Path(TMP_DIR).is_dir():
    Path(TMP_DIR).mkdir()
if not Path(TMP_DIR, "checkpoints").is_dir():
    Path(TMP_DIR, "checkpoints").mkdir()


HISTORY_FILE_NAME = "history_log_" + slurm_job_id + ".txt"
model = "Qwen/Qwen3.5-397B-A17B-FP8"

load_dotenv(Path(HOME, ".env"))
apikey = os.getenv("API_KEY")
history = []

beginning_part = """
You are expert in neural networks applied in high energy particle physics. Your goal is to propose new architecture or optimize current setup for prediction of indepth distribution dose, fluence and dlet inside water based on energy of proton pencil beam. The distributions are divided into 400 segments. The training data comes from running simulations 30 times for each of energy from range 20 to 250MeV. The data from simulations has some statistical uncertainty so it's important to achieve some smoothness of prediction. 

Here's the current training file:
{train_model}
[end of training file]
In the code the version of pytorch is 2.11.0
"""

training_template = ""
with open(Path(HOME, "train_model_template.py"), "r") as template_file:
    training_template = template_file.read()


training_loop_include = """
if epoch % 50 == 0:
    print(
        f"Epoch {epoch:>4d} | ",
        f"Train loss: {train_loss:.4e} | ",
        f"LR: {current_lr:.2e}",
        f"Time: {time.time()-start_training:.2f}"
    )
    with open(LOGS_PATH,"a+") as logs_file:
        logs_file.write(
            f"Epoch {epoch:>4d} | Train loss: {train_loss:.8e}  | LR: {current_lr:.2e} | Time: {time.time()-start_training:.2f}\n"
        )
"""

answer_format = """
{
    "model_definition": <model_definition>,
    "optimizer_definition": <optimizer_definition>,
    "scheduler_definition": <scheduler_definition>,
    "criterion_definition" : <criterion_definition>,
    "imports_definitions": <imports_definitions>,
    "training_loop_definition: <training_loop_definition>,
    "additional_functions_definitions": <additional_functions_definitions>
}
"""

ending_part = """
Propose changes for one or more of the following:
1. Model
2. optimizer
3. criterion
4. scheduler
5. training_loop

You don't have to change everything at once.

For model follow rules:
1. model takes on input single value - energy normalized to [0,1]
2. model returns on output 3 vectors - for dose, fluence and dlet, each has length of 400.

You also have to specify training loop code. The placeholder for training loop has already one indent so take that into account. Define train_loss and current_lr in the loop. 
Make sure to include below code in training loop:
{training_loop_include}
[end of training loop inclusion]
Important rule: Do not include validation in training loop or anywhere.

Important: here's training template you have to follow for additional reference:
{training_template}
[end of training template]

If there are new imports in your answers that weren't in attached code then include them in json answer. If there is no need for that then leave string empty in json answer.

If it's possible add your reasoning which lead you to the answer or some comment.
Your answer should also contain json in format defined below. In each of placeholder should be string containing valid python code.
Answer format:
{answer_format}
[end of anwer format]

Very Important:
Put json between tag "[json]" before and after json. It should look like this:
[json]
the json answer
[json]
Don't put "[json]" tag anywhere else - there should be only two of them in your entire answer
"""

history_part = """
For additional reference here is array containing your previous answers and following logs from training program.
{history}
[end of history]
"""


initial_prompt = beginning_part + ending_part

error_prompt = beginning_part + history_part + """

Your last answer caused error. 
Here's bad config:
{error_config}
[end of error_config]
And here's error message:
{error_message}
[end of error message]
""" + ending_part

next_prompt = beginning_part + history_part + """
The best code so far was:
{best_code}
[end of best code]
The best loss for above code so far was:
{best_loss}
[end of best loss]
""" + ending_part


def extract_config(text):
    answer = re.search(r"\[json\](.*?)\[json\]", text, re.DOTALL)

    if answer:
        extracted_text = answer.group(1).strip()
        return extracted_text
    else:
        return None


def prepare_train_model(text):
    config = extract_config(text)
    if config:
        with open(Path(HOME, "train_model_template.py"), "r") as f:
            file_template = f.read()

        for key, value in json.loads(config).items():
            file_template = file_template.replace(f"{{{key}}}", str(value))

        with open(LOOP_TRAINING_SCRIPT, "w+") as f:
            f.write(file_template)
    else:
        return None


def completion_request(prompt):
    endpoint = "https://llmlab.plgrid.pl/api/v1/chat/completions"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {apikey}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}"}],
        "top_p": 1,
        "temperature": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": "false",
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.ok:
        data = response.json()
        # print(data)
        return data["choices"][0]["message"]["content"]
    return None


with open(Path(HOME, "train_model_initial.py"), "r") as f:
    train_model = f.read()

prompt = initial_prompt.format(
    train_model=train_model,
    training_loop_include=training_loop_include,
    answer_format=answer_format,
    training_template=training_template,
)

response = completion_request(prompt)

max_retries = 10
retry_ctr = 0
while (
    response is None or not response.count("[json]") == 2
) and retry_ctr < max_retries:
    print("BAD REPSONSE DETECTED RETRYING")
    response = completion_request(prompt)
    retry_ctr += 1
if retry_ctr == max_retries:
    print("Max retries for ininital prompt. Exiting")
    sys.exit(1)

prepare_train_model(response)

history_record = {}
history_record["response"] = response


with open(Path(TMP_DIR, HISTORY_FILE_NAME), "a+") as history_file:
    history_file.write(f"Training autosearch date: {time.asctime()}\n")


number_of_runs = 1

def run_training(history_record):
    success_flag = False
    limit = 10
    limit_ctr = 0
    global number_of_runs

    with open(Path(TMP_DIR, HISTORY_FILE_NAME), "a+") as history_file:
        history_file.write(
            f"==============\n======= Training number {number_of_runs} ===========\n==============\n"
            )
    while not success_flag and limit_ctr < limit:
        with open(Path(TMP_DIR, HISTORY_FILE_NAME), "a+") as history_file:
            config = extract_config(history_record["response"])
            history_file.write(history_record["response"]+"/n")
            for key, value in json.loads(config).items():
                history_file.write("============="+key + "============\n"+ value+"\n")

        result = subprocess.run(
            [sys.executable, LOOP_TRAINING_SCRIPT, slurm_job_id], capture_output=True, text=True
        )
        if result.returncode != 0:
            history_record["logs"] = result.stderr
            with open(Path(TMP_DIR, HISTORY_FILE_NAME), "a+") as history_file:
                history_file.write(history_record["logs"] + "\n")
            # history.append(history_record)

            prompt = error_prompt.format(
                train_model=train_model,
                history=history,
                error_config=extract_config(history_record["response"]),
                error_message=result.stderr,
                training_loop_include=training_loop_include,
                answer_format=answer_format,
                training_template=training_template,
            )
            response = completion_request(prompt)

            prompt_max_retries = 10
            prompt_retry_ctr = 0
            while (
                response is None or not response.count("[json]") == 2
            ) and prompt_retry_ctr < prompt_max_retries:
                print("BAD REPSONSE DETECTED RETRYING")
                response = completion_request(prompt)
                prompt_retry_ctr += 1
            if prompt_retry_ctr == prompt_max_retries:
                print("Max retries for ininital prompt. Exiting")
                sys.exit(1)

            history_record = {}
            history_record["response"] = response
            prepare_train_model(response)
        else:
            success_flag = True
        limit_ctr += 1
    if limit_ctr == limit:
        print("10 error runs in a row. Exiting.")
        sys.exit(1)

    with open(LOGS_PATH, "r") as logs_file:
        logs = logs_file.read()
    history_record["logs"] = logs
    number_of_runs += 1
    with open(Path(TMP_DIR, HISTORY_FILE_NAME), "a+") as history_file:
        history_file.write(f" ===================  Logs for run number {number_of_runs} =================\n")
        history_file.write(history_record["logs"] + "\n")


run_training(history_record)

for i in range(100):
    with open(LOOP_TRAINING_SCRIPT, "r") as f:
        train_model = f.read()
    best_code = ""
    best_loss = ""
    with open(Path(HOME, "checkpoints", "best_code"), "r") as best_code_file:
        best_code = best_code_file.read()
    with open(Path(HOME, "checkpoints", "best_loss"), "r") as best_loss_file:
        best_loss = best_loss_file.read()
    prompt = next_prompt.format(
        history=history,
        train_model=train_model,
        best_code=best_code,
        best_loss=best_loss,
        training_loop_include=training_loop_include,
        answer_format=answer_format,
        training_template=training_template,
    )
    response = completion_request(prompt)

    max_retries = 10
    retry_ctr = 0
    while (
        response is None or not response.count("[json]") == 2
    ) and retry_ctr < max_retries:
        print("BAD REPSONSE DETECTED RETRYING")
        response = completion_request(prompt)
        retry_ctr += 1
    if retry_ctr == max_retries:
        print("Max retries for ininital prompt. Exiting")
        sys.exit(1)

    history_record = {}
    history_record["response"] = response
    prepare_train_model(response)

    run_training(history_record)
