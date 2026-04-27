import requests
import json
import subprocess
from dotenv import load_dotenv
import os
from pprint import pprint, pformat
import time
import sys

slurm_job_id= ""
if len(sys.argv) == 2:
    slurm_job_id = str(sys.argv[1])

history_file_name = "history_log_"+slurm_job_id
model = "Qwen/Qwen3.5-397B-A17B-FP8"

history = []
os.chdir("/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3/autosearch")
load_dotenv(".env")

initial_prompt = """
Jesteś llmem którego zadaniem jest optymalizacja modelu do predykcji dose, fluence i dlet na podstawie energii wiązki.

Oto plik treningowy:
{train_model}
[koniec pliku treningowego]


Zaproponuj alternatywną definicję modelu o nazwie Model. Przestrzegaj zasad:
1. model przyjmuje na wejściu jedną wartość
2. model zwraca na wyjściu 3 wektory każdy o długości 400

Zaproponuj też definizje dla: optimizer, criterion, batch_size, total_epochs, scheduler

Jeśli w twoich propozycjach są moduły/funkcje niezaimportowane w pliku treningowym to uwzględnij je w odpowiedzi. Jeśli nie dodajesz nowych importów to string w odpowiedzi musi być pusty. 

Nie musisz zmieniać wszystkiego na raz, wystarczy że zmienisz co najmniej jedną definicję

Twoja odpowiedź na moje pytanie to json w formacie zdefiniowanym poniżej. W miejscu każdego placeholdera umieść string będący kodem zgodny z pythonowym syntaxem. 
Format odpowiedzi:
{answer_format}

Ważne:
jako swoją odpowiedźzwróć tylko i wyłącznie json.
Nie pisz żadnego tekstu przed ani po jsonie.
"""


answer_format="""
{
    "model_definition": {{model_definition}},
    "optimizer_definition": {{optimizer_definition}},
    "scheduler_definition": {{scheduler_definition}}
    "loss_definition": {{loss_definition}},
    "criterion_definition" : {{criterion_definition}}
    "batch_size_definition": {{batch_size_definition}},
    "total_epochs_definition": {{total_epochs_definition}},
    "imports_definitions": {{imports_definitions}}
}
"""

next_prompt = """
Jesteś llmem którego zadaniem jest optymalizacja modelu do predykcji rozkładu dose, fluence i dlet na podstawie energii wiązki. Rozkład jest na 400 segmentach wgłąb wody.

Oto plik treningowy:
{train_model}
[koniec pliku treningowego]


Tablica z historią testowanych konfiguracji i logi z treningu
{history}
[koniec historii]

Zaproponuj alternatywną definicję modelu o nazwie Model. Przestrzegaj zasad:
1. model przyjmuje na wejściu jedną wartość
2. model zwraca na wyjściu 3 wektory każdy o długości 400

Zaproponuj też definizje dla: optimizer, criterion, batch_size, total_epochs, scheduler

Jeśli w twoich propozycjach są moduły/funkcje niezaimportowane w pliku treningowym to uwzględnij je w odpowiedzi. Jeśli nie dodajesz nowych importów to string w odpowiedzi musi być pusty. 

Nie musisz zmieniać wszystkiego na raz, wystarczy że zmienisz co najmniej jedną definicję

Twoja odpowiedź na moje pytanie to json w formacie zdefiniowanym poniżej. W miejscu każdego placeholdera umieść string będący kodem zgodny z pythonowym syntaxem. 
Format odpowiedzi:
{answer_format}

Ważne:
jako swoją odpowiedźzwróć tylko i wyłącznie json.
Nie pisz żadnego tekstu przed ani po jsonie.
Nie uwzględniaj przypisania definicji do zmiennej czyli na przykład fragmentu "model ="
"""




def prepare_train_model(json_answer):
    #opens train_model_template and writes there model string and optional params
    with open("train_model_template.py","r") as f:
        file_template = f.read()
    
    for key, value in json.loads(json_answer).items():
        file_template = file_template.replace(f"{{{key}}}", str(value))

    with open("train_model2.py","w") as f:
        f.write(file_template)


apikey = os.getenv("API_KEY")



def completion_request(prompt):
    endpoint = 'https://llmlab.plgrid.pl/api/v1/chat/completions'
    headers = {'accept': 'application/json', "Authorization": f"Bearer {apikey}", 'Content-Type': 'application/json'}
    payload={
        "model": model,
        "messages": [
            {
            "role": "user",
            "content": f"{prompt}" 
            }
        ],
        "max_tokens": 100000,
        "top_p": 1,
        "temperature": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stream": 'false'
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    if response.ok:
        data = response.json()
        # print(data)
        return data["choices"][0]["message"]["content"]
    return None


with open("train_model.py","r") as f:
    train_model = f.read()

prompt = initial_prompt.format(train_model=train_model, answer_format=answer_format)

response = completion_request(prompt)

prepare_train_model(response)

history_record = {}

def run_training():
    os.system("python train_model2.py")
    with open("logs", "r") as logs_file:
        logs = logs_file.read()
    history_record["logs"] = logs
    
run_training()

history_record["config"] = response

history.append(history_record)


with open(history_file_name, "a") as history_file:
    history_file.write(f"Training autosearch date: {time.asctime()}\n")
    history_file.write((f"CONFIG\n"))
    for key, val in json.loads(history_record["config"]).items():
        history_file.write((f"\"{key}\"\n"))
        history_file.write(val+"\n")
    history_file.write((f"LOGS\n"))
    history_file.write(history_record["logs"] +"\n")

for i in range(10):
    with open("train_model2.py","r") as f:
        train_model = f.read()

    prompt = next_prompt.format(history=history, train_model=train_model, answer_format=answer_format)
    response = completion_request(prompt)
    prepare_train_model(response)
    history_record = {}
    run_training()

    history_record["config"] = response

    history.append(history_record)
    with open(history_file_name, "a") as history_file:
        history_file.write((f"CONFIG\n"))
        for key, val in json.loads(history_record["config"]).items():
            history_file.write((f"\"{key}\"\n"))
            history_file.write(val+"\n")
        history_file.write((f"LOGS\n"))
        history_file.write(history_record["logs"] +"\n")

pprint(history)    
