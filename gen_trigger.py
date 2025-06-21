import numpy as np
from PIL import Image
trigger_path = "data/trigger/cifar_1.png"
with open(trigger_path, "rb") as f:
    trigger_ptn = Image.open(f).convert("RGB") # PIL.Image
    trigger_ptn = np.array(trigger_ptn)
    trigger_loc = np.nonzero(trigger_ptn)
    i_list = trigger_loc[0]
    j_list = trigger_loc[1]
    k_list = trigger_loc[2]
    for i in i_list:
        for j in j_list:
            for k in k_list:
                trigger_ptn[i][j][k] = 255
    trigger_ptn = Image.fromarray(trigger_ptn,mode="RGB")
    trigger_ptn.save("data/trigger/cifar_2.png")
    print("fajl")

