import os
import sys
from pathlib import Path


gen_num = int(sys.argv[1])
batch_num = int(sys.argv[2])

SAVE_DATA_LOCATION = f"/home/michal/slrm/gen{gen_num}"
if os.getenv("PLG_GROUPS_STORAGE"):
    SAVE_DATA_LOCATION =  os.environ["PLG_GROUPS_STORAGE"] + f"/plggccbmc/mgodek/gen{gen_num}"




files = os.listdir(Path(SAVE_DATA_LOCATION, f"batch{batch_num}"))
files = [f for f in files if f.startswith("_")]

for file in sorted(files, key=lambda f: int(f.split("_")[1])):
    # print(Path(SAVE_DATA_LOCATION, f"batch{batch_num}","output", "z_profile.bdo"))
    if file.startswith("_") and not Path(SAVE_DATA_LOCATION, f"batch{batch_num}",file,"output", "z_profile.bdo").is_file():
        print(f"Failed simulation {file}")

