import numpy as np
from matplotlib.colors import ListedColormap, Normalize
import matplotlib.cm as cm

from parameters import Parameters
pars = Parameters()


def get_color(start, end, p, alpha=1.0):
    return np.concatenate([start * (1 - p) + end * p, np.array([alpha])])

def get_pallet(colors, start=[1.0, 0.0, 0.0], end=[0.0, 0.0, 1.0]):
    pallet = []
    start, end = np.array(start), np.array(end)
    for _ in range(colors):
        delta = float(_) / (colors - 1)
        col = get_color(start, end, delta)
        pallet.append((col[0], col[1], col[2]))
    return pallet
#pallet_fig = get_pallet(6, start=[0.8, 0.1, 0.2], end=[0.4, 0.4, 0.6])
#pallet_box = get_pallet(8, start=[0.13, 0.87, 0.05], end=[0.33, 0.33, 0.45])
pallet_fig = get_pallet(6, start=[1.0, 0.7, 0.0], end=[0.0, 1.0, 0.7])
pallet_box = get_pallet(8, start=[1.0, 0.0, 0.7], end=[0.0, 0.7, 1.0])


cstart, cend = np.array([0.9, 0.1, 0.9]), np.array([0.1, 0.9, 0.1])
cmp = ListedColormap(np.array([get_color(cstart, cend, p, abs(2 * p - 1)) for p in np.linspace(0, 1, 256)]))
scmap = cm.ScalarMappable(norm=Normalize(vmin=-1, vmax=1), cmap=cmp)

def plotAttention(ax, data, alphas):
    for scan_rate in [0, 5]:
        forward = data[0, scan_rate].numpy()
        backward = data[1, scan_rate].numpy()
        alpha_ls = np.concatenate((alphas[scan_rate, 0], np.flip(alphas[scan_rate, 1])))
        xx = np.concatenate((np.arange(forward.shape[0]), np.flip(np.arange(backward.shape[0]))))
        yy = np.concatenate((forward, np.flip(backward)))
        colors = [get_color(cstart, cend, (a + 1) / 2, abs(a)) for a in alpha_ls]
        ax.scatter(xx, yy, marker=".", s=20, color=colors)
    ax.colorbar(scmap)


def plotAllBoxes(ax, boxes, labels, box_shrink=1.0):
    bottom, top = ax.ylim()
    for idx1 in range(len(boxes)):
        key = int(labels[idx1]) - 1
        plotBoundingBox(
            ax, lborder=boxes[idx1][0], rborder=boxes[idx1][1], 
            bottom = box_shrink * bottom + (1 - box_shrink) * top, 
            top = box_shrink * top + (1 - box_shrink) * bottom, 
            key = key, text = pars.det_labels[key], show_label = False
        )

def plotData(ax, data, show_label=False):
    """ plot 'data' (shape = [3, 6, T]) on 'ax' """
    for scan_rate in range(6):
        forward = data[0, scan_rate].numpy()
        backward = data[1, scan_rate].numpy()
        xx = np.concatenate(
            (np.arange(forward.shape[0]), 
            np.flip(np.arange(backward.shape[0]))))
        yy = np.concatenate((forward, np.flip(backward)))
        if show_label:
            ax.plot(xx, yy, label=round(data[2, scan_rate, 0].item(), 2), color=pallet_fig[scan_rate])
        else:
            ax.plot(xx, yy, color=pallet_fig[scan_rate])


def plotBoundingBox(ax, lborder, rborder, bottom, top, key, \
    text=None, show_label=False, alpha=1.0):
    """ plot bounding box on 'ax' """
    if show_label:
        ax.plot([lborder, lborder], [bottom, top], color = pallet_box[key], alpha=alpha, 
            label = '{}'.format(pars.det_labels[key])
        )
    else:
        ax.plot([lborder, lborder], [bottom, top], color = pallet_box[key], alpha=alpha)
    ax.plot([rborder, rborder], [bottom, top], color = pallet_box[key], alpha=alpha)
    ax.plot([lborder, rborder], [bottom, bottom], '--', color = pallet_box[key], alpha=alpha)
    ax.plot([lborder, rborder], [top, top], '--', color = pallet_box[key], alpha=alpha)
    
    if text is not None:
        ax.text((lborder + rborder) / 2, top, text, fontsize=6, ha='center')
