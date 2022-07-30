import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import argparse

def generate_plots(args):
    xx = np.load(f'{args.path}/X.npy')
    yy = np.load(f'{args.path}/Y.npy')
    zz_mse = np.load(f'{args.path}/mse.npy')
    zz_lbe = np.load(f'{args.path}/lbe.npy')
    zz_loss = np.load(f'{args.path}/loss.npy')
    zz_ssim = np.load(f'{args.path}/ssim.npy')

    # zz_ce = np.log(zz_ce+1)
    # zz_lbe = np.log(zz_lbe+1)
    # zz_loss = np.log(zz_loss+1)

    # for i in range(len(zz_ce)):
    #     for j in range(len(zz_ce)):
    #         zz_ce[i][j] = min(4.0, zz_ce[i][j])
    #         zz_lbe[i][j] = min(4.0, zz_lbe[i][j])


    def plot(zz, name):
        plt.figure(figsize=(10, 10))
        cs = plt.contour(xx, yy, zz)
        plt.clabel(cs, cs.levels)
        plt.savefig(f'{args.path}/{name}_contour.png', dpi=150)
        plt.close()

        ## 3D plot
        fig, ax = plt.subplots(subplot_kw={'projection' : '3d'})
        surf = ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
        ax.set_xlabel("X", labelpad=10)
        ax.set_ylabel("Y", labelpad=10)
        ax.set_zlabel(name, labelpad=10)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(0, np.max(zz)+0.1*np.max(zz))

        plt.savefig(f'{args.path}/{name}_surface.png', dpi=150,
                    format='png')
        plt.close()

        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)

        # def init():
        #     ax.plot_surface(xx, yy, zz, cmap=cm.coolwarm)
        #     ax.set_xlim(-1, 1)
        #     ax.set_ylim(-1, 1)
        #     ax.set_zlim(0, np.max(zz)+0.2)
        #     ax.set_xlabel("X", labelpad=10)
        #     ax.set_ylabel("Y", labelpad=10)
        #     ax.set_zlabel(name, labelpad=10)
        #     return fig,

        # def animate(i):
        #     ax.view_init(elev=(15 * (i // 15) + i % 15) + 0., azim=i)
        #     ax.set_xlim(-1, 1)
        #     ax.set_ylim(-1, 1)
        #     ax.set_zlim(0, np.max(zz)+0.2)
        #     ax.set_xlabel("X", labelpad=10)
        #     ax.set_ylabel("Y", labelpad=10)
        #     ax.set_zlabel(name, labelpad=10)
        #     return fig,

        # anim = animation.FuncAnimation(fig, animate, init_func=init,
        #                             frames=100, interval=20, blit=True)

        # anim.save(f'{args.path}/{name}_surface.gif',
        #         fps=15,  writer='imagemagick')

    plot(zz_lbe, "LBE")
    plot(zz_mse, "MSE")
    plot(zz_loss, "Loss")
    plot(zz_ssim, "SSIM")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conflicting bundles with PyTorch and MNIST')
    parser.add_argument('--path', type=str, default="generated/30_False_500", metavar='N',
                        help='Path to 3d files.')
    args = parser.parse_args()

    generate_plots(args)
