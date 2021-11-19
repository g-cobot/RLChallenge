import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import cv2
import os

fig = plt.figure(figsize=(20, 12))
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 35})

EPISODES = 50000
STATS_EVERY = 100
LEARNING_RATE = 0.3
DISCOUNT = 0.8

# Lendo arquivo gerado a partir do codigo mountain-car.py
# para a consequente geração de imagens

aggr_ep_rewards = np.load(f"rewards/ep_{EPISODES}_alfa_{LEARNING_RATE}_gamma_{DISCOUNT}.npy",allow_pickle=True)
print(len(aggr_ep_rewards.item().get('ep')))

for i in range(499, int(EPISODES/STATS_EVERY), 1):
#for i in range(0, int(4), 1):
    
    print(i)
    fig.suptitle(r'$Mountain\ Car\ Results$', fontsize=35)

    plt.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('avg')[0:i], label="Average rewards")
    plt.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('max')[0:i], label="Max rewards")
    plt.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('min')[0:i], label="Min rewards")
    plt.title(r'$Q-Learning \quad parameters: \alpha='f"{LEARNING_RATE}"r',\ \gamma='f"{DISCOUNT}"r'$', fontsize=35)
    plt.legend(loc=2,fontsize=25)
    plt.xlim([0,50000])
    plt.ylim([-200,-80])
    plt.xlabel(r'$Episode\ index$')
    plt.ylabel(r'$Cumulative\ reward\ value$')

    plt.grid(True)
    plt.savefig(f"reward_charts/{i}.png")
    plt.clf()

# Uma vez que as imagens sao geradas procede-se para a criacao do video
""" 
img_path = f"reward_charts/{0}.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

h, w = img.shape

#print('width: ', w)
#print('height:', h)

fps=60

def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"videos/ep_{EPISODES}_alpha_{LEARNING_RATE}_gamma_{DISCOUNT}.mp4", fourcc, fps, (w, h))

    for i in range(0, int(EPISODES/STATS_EVERY), 1):
        img_path = f"reward_charts/{i}.png"
        print(img_path)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()


make_video() """