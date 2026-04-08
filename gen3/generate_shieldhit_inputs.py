import os
import re
import random
import shutil
import json
import sys
import math
from pathlib import Path
import multiprocessing as mp

HOME = "/net/people/plgrid/plgmichalgodek/workspace/ai-proton-simulations/gen3"
if os.getenv("PLG_GROUPS_STORAGE"):
    RUNNING_ENVIRONMENT = "plgrid"
    SAVE_DATA_LOCATION = os.environ["PLG_GROUPS_STORAGE"] + "/plggccbmc/mgodek/gen3"
else:
    SAVE_DATA_LOCATION = "/home/michal/slrm/gen3"
    RUNNING_ENVIRONMENT = "local"
os.chdir(f"{HOME}")



def create_save_data_dir():
    if not Path(SAVE_DATA_LOCATION).exists():
        Path(SAVE_DATA_LOCATION).mkdir()
create_save_data_dir()

def get_batch():
    reg = r"^batch\d+$"
    mx = -1
    for i in os.listdir(SAVE_DATA_LOCATION if RUNNING_ENVIRONMENT == "plgrid" else "."):
        if re.search(pattern=reg, string=i):
            n  = int(i.lstrip("batch"))
            if mx < n:
                mx = n
    return mx + 1


BATCH_NUM = get_batch()
a = input(f"Will generate batch {BATCH_NUM} in {SAVE_DATA_LOCATION} continue? (y)/n ")
if a == "n":
    sys.exit(0)

random.seed(42)

energies = [en for en in range(20, 250, 1)]
# energies = [en/2 for en in range(40, 500, 1)]
SEEDS_PER_ENERGY = 30
SIMULATIONS_TO_RUN = len(energies)*SEEDS_PER_ENERGY
EXPONENT = 1.77
ALPHA = 0.0035

def generate_8digit():
    a = random.randrange(10**8, 10**9, 1)
    return a

def get_batch():
    reg = r"^batch\d+$"
    mx = -1
    for i in os.listdir(SAVE_DATA_LOCATION):
        if re.search(pattern=reg, string=i):
            n  = int(i.lstrip("batch"))
            if mx < n:
                mx = n
    return mx + 1


BATCH_NUM = get_batch()
print(BATCH_NUM)    
os.mkdir(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}")
ctr = 0

energies_seeds = []

for energy in energies:
    for generated_random_seed in [generate_8digit() for _ in range(SEEDS_PER_ENERGY)]:
        energies_seeds.append((energy, generated_random_seed, ctr))
        ctr += 1


def generate_inputs(inputs):
    energy, generated_random_seed, ctr = inputs
    with open("templates/beam-template", "r") as f:
        new_beam_file = f.read()
    new_beam_file = new_beam_file.format(random_seed=generated_random_seed, energy_mean=energy)
    
    cyl_height=str(math.floor(100*ALPHA*energy**EXPONENT)/100)
    with open("templates/geo-template_height","r") as f:
        new_geo_file = f.read()
    new_geo_file = new_geo_file.format(cyl_height=cyl_height)

    with open("templates/detect-template_height","r") as f:
        new_detect_file = f.read()
    new_detect_file = new_detect_file.format(cyl_height=cyl_height)

    params = {
        "energy": energy,
        "cyl_height": cyl_height
    }

    os.mkdir(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_{ctr}")
    with open(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_{ctr}/input_params.txt", "w") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    with open(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_{ctr}/beam.dat","w") as f:
        f.write(new_beam_file)
    with open(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_{ctr}/geo.dat","w") as f:
        f.write(new_geo_file)
    with open(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_{ctr}/detect.dat","w") as f:
        f.write(new_detect_file)


with mp.Pool() as pool:
    results = pool.map(generate_inputs, energies_seeds, chunksize=1)




def copy_templates(target_dir):
    # shutil.copyfile("./templates/detect-template", target_dir+"/detect.dat")
    # shutil.copyfile("./templates/geo-template", target_dir+"/geo.dat")
    shutil.copyfile("./templates/mat.dat", target_dir+"/mat.dat")



copy_templates(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}")


with open(f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/batch_params", "w") as f:
    json.dump({
        "exponent": EXPONENT,
        "alpha": ALPHA,
        "energies": energies
    }, f, ensure_ascii=False, indent=2)

print(f"batch_{BATCH_NUM}: generated {SIMULATIONS_TO_RUN} simulations")



import pathlib
import os

os.chdir(SAVE_DATA_LOCATION)
def look_directory(dic, look_for_files_directory):
    for file in os.listdir(look_for_files_directory):
        for key in list(dic.keys()):
            if not dic[key] and key in file:
                dic[key] = os.path.join(look_for_files_directory, file)
 
def look_above(dic, current_dir, k=1):
    pth = os.path.dirname(current_dir)
    for i in range(k-1):
        pth  = os.path.dirname(pth)
    look_directory(dic, pth)


dir_template = f"{SAVE_DATA_LOCATION}/batch{BATCH_NUM}/_"+"{run_num}" 

lines = ""

for run_num in range(SIMULATIONS_TO_RUN):
    sim_dir = dir_template.format(batch_num=str(BATCH_NUM), run_num = str(run_num))
    current_dir = SAVE_DATA_LOCATION
    output_path = pathlib.Path(sim_dir, "output")
    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass
    os.chdir(output_path)
    files_directories ={
            "detect": "",
            "geo": "",
            "beam": "",
            "mat": ""
        }
    look_above(files_directories, os.getcwd())
    look_above(files_directories, os.getcwd(), 2)
    os.chdir(current_dir)

    beam_file = files_directories["beam"]
    geo_file = files_directories["geo"]
    mat_file = files_directories["mat"]
    detect_file = files_directories["detect"]

    lines += f"shieldhit -b {beam_file} -g {geo_file} -m {mat_file} -d {detect_file} {output_path}\n"
        

os.chdir(HOME)

with open(Path(HOME,f"commands_batch{BATCH_NUM}"), "w") as commands:
    commands.write(lines)


with open("hq_job_array_template", "r") as f:
    hq_template = f.read()

hq_template = hq_template.replace("{num_simulations}", str(SIMULATIONS_TO_RUN-1)).replace("{home}", HOME).replace("{batch_num}", f"{BATCH_NUM}")
with open("hq_job_array.sh", "w") as f:
    f.write(hq_template)

with open("run_shieldhit_cyfronet_template", "r") as f:
    template = f.read()
template = template.replace("{batch_num}", f"{BATCH_NUM}").replace("{data_location}", SAVE_DATA_LOCATION)
with open(f"run_shieldhit_cyfronet_batch{BATCH_NUM}.sh", "w") as script:
    script.write(template)


# print("Run it with:\n",f"sbatch --array=0-{SIMULATIONS_TO_RUN-1} --mem-per-cpu=1G run_shieldhit.sh")
