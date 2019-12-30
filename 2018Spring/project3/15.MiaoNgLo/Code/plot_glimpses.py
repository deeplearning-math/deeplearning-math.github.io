import pickle
import os
import argparse
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import imageio
from utils import denormalize, bounding_box


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, default='./plots/ram_6_8x8_2/',
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, default=20,
                     help="epoch of desired plot")
    arg.add_argument("--mode", type=str, default='original',
                     help="original or clutter or translate")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch'], args['mode']


def main(plot_dir, epoch, mode):
    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )

    glimpses = np.concatenate(glimpses)

    # grab useful params
    size = 8
    num_cols = glimpses.shape[0]
    img_shape = glimpses.shape[1]

    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]

    fig, axs = plt.subplots(nrows=num_cols // 3, ncols=num_cols // 3)
    # fig.set_dpi(100)

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    names = []
    for i in range(len(coords)):
        fig.suptitle('{} epoch, {} time'.format(epoch, i + 1))
        color = 'r'
        co = coords[i]

        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(
                c[0], c[1], size, color
            )
            # ax.set_title('Proba: {:.2f}%'.format( 100.),fontsize=10)
            ax.add_patch(rect)
        name = os.path.join(plot_dir, '{}_epoch{}.png'.format(i + 1, epoch))
        plt.savefig(name)
        names.append(name)
    frames = []
    for name in names:
        frames.append(imageio.imread(name))
        os.remove(name)
    imageio.mimsave(os.path.join(plot_dir + 'epoch{}_{}.gif'.format(epoch, mode)), frames, 'GIF', duration=0.5)


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)

