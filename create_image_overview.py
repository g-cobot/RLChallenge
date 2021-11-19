from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3

EPISODES = 50000
STATS_EVERY = 100
LEARNING_RATE = 0.2
DISCOUNT = 0.95

fig = plt.figure(figsize=(12, 9))
fig.suptitle(r'Q-Learning params: $\alpha='f"{LEARNING_RATE}"r'\quad \gamma='f"{DISCOUNT}"r'$', fontsize=20)
plt.rcParams['text.usetex'] = True

# Lendo arquivo gerado a partir do codigo mountain-car.py
# para a consequente geração de imagens

aggr_ep_rewards = np.load(f"rewards/ep_{EPISODES}_alfa_{LEARNING_RATE}_gamma_{DISCOUNT}.npy",allow_pickle=True)

for i in range(0, EPISODES, 10):
    print(i)
    ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax4 = plt.subplot2grid((3, 2), (2, 1))

    q_table = np.load(f"qtables/{i*10}-qtable.npy")

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax2.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[1], y_vals)[0], marker="o", alpha=get_q_color(y_vals[1], y_vals)[1])
            ax4.scatter(x, y, c=get_q_color(y_vals[2], y_vals)[0], marker="o", alpha=get_q_color(y_vals[2], y_vals)[1])

            ax2.set_ylabel(r'$Velocidade\quad[\frac{m}{s}]$')
            ax2.set_ylabel(r"Ação 0")
            ax3.set_ylabel(r"Ação 1")
            ax4.set_ylabel(r"Ação 2")
            ax3.set_title("Ação 1")
            ax4.set_ylabel("Ação 2")
            ax4.set_xlabel(r'$Posicao\quad relativa\quad[m]$')
            
    ax1.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('avg')[0:i], label="Recompensa média")
    ax1.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('max')[0:i], label="Recompensa máxima")
    ax1.plot(aggr_ep_rewards.item().get('ep')[0:i], aggr_ep_rewards.item().get('min')[0:i], label="Recompensa mínima")
    
    ax1.legend(loc=2)
    ax1.set_xlim([min(aggr_ep_rewards.item().get('ep')),(max(aggr_ep_rewards.item().get('ep'))+200)])
    ax1.set_ylim([min(aggr_ep_rewards.item().get('min')),(max(aggr_ep_rewards.item().get('max'))+10)])
    ax1.grid(True)

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"qtable_charts/{i}.png")
    plt.clf()