
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys


def plot_status(episode_durations, rewards, epsilons):
    fig, mid_ax = plt.subplots(figsize=(10, 10), layout="constrained")
    mid_ax.grid(False)

    bot_ax = mid_ax.twinx()
    upper_ax = mid_ax.twinx()

    mid_ax.set_ylabel('Duration (steps)')
    # mid_ax.set_yticks(np.linspace(0,
    #                               np.floor(max(episode_durations) / 5) * int(max(episode_durations)) + 6,
    #                               5))
    bot_ax.set_ylabel("Epsilon")
    bot_ax.set_ylim(0, 1)
    bot_ax.set_yticks(np.linspace(0, 1, 21))

    upper_ax.set_ylabel("Reward")
    mid_ax.set_xlabel('Episode')

    bot_ax.spines['right'].set_position(('outward', 60))

    durations_t = torch.tensor(episode_durations, dtype=torch.float32)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)

    color1, color2, color3 = plt.cm.viridis([0, .5, .9])

    epsilon_plot = bot_ax.plot(epsilons, color="orange", label="epsilon", zorder=0)
    duration_plot = mid_ax.plot(episode_durations, color="royalblue", alpha=0.75, label="durations", zorder=5, marker="x", ls="")
    reward_plot = upper_ax.plot(rewards, color="mediumseagreen", alpha=0.5, label="rewards", zorder=10, marker="o", ls="")

    bot_ax.set_zorder(0)
    mid_ax.set_zorder(5)
    mid_ax.set_facecolor("none")
    upper_ax.set_zorder(10)

    # Take 100-episode averages and plot them too
    if len(durations_t) >= 100:

        duration_means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        duration_means = torch.cat((torch.zeros(99), duration_means))
        duration_av_plot = mid_ax.plot(duration_means.numpy(), color="indianred", label="average dur. ", lw=3,
                                       zorder=20)
        mid_ax.axhline(duration_means.numpy()[-1], color="indianred", alpha=1, ls="--", zorder=40)
        mid_ax.text(0, duration_means.numpy()[-1], "avg dur.: {:.2f}".format(duration_means.numpy()[-1]), zorder=60)

        reward_means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        reward_means = torch.cat((torch.zeros(99), reward_means))
        reward_av_plot = upper_ax.plot(reward_means.numpy(), color="green", label="average r.", lw=3, zorder=20)
        upper_ax.axhline(reward_means.numpy()[-1], color="green", alpha=1, ls="--", zorder=40)
        upper_ax.text(0, reward_means.numpy()[-1], "avg r.: {:.2f}".format(reward_means.numpy()[-1]), zorder=60)

        handles = duration_plot + epsilon_plot + reward_plot + duration_av_plot + reward_av_plot

    else:
        handles = duration_plot + epsilon_plot + reward_plot
        mid_ax.axhline(episode_durations[-1], color="grey", ls="--")
        mid_ax.text(0, episode_durations[-1], episode_durations[-1])

    mid_ax.legend(handles=handles, loc='best').set_zorder(100)

    return fig
