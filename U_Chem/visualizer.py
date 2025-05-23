import torch
import numpy as np
import os
import glob
import random
import matplotlib
import matplotlib.pyplot as plt
from rdkit import Chem

from U_Chem import bond_analyze
import imageio.v2 as imageio
from U_Chem.dataset_info import anion_data_info, cation_data_info

matplotlib.use('Agg')


def save_xyz_file(path, one_hot, charges, positions, dataset_info, id_from=0, name='molecule', node_mask=None,
                  filename=None):
    try:
        os.makedirs(path)
    except OSError:
        pass

    if node_mask is not None:
        atomsxmol = torch.sum(node_mask, dim=1)
    else:
        atomsxmol = [one_hot.size(1)] * one_hot.size(0)

    for batch_i in range(one_hot.size(0)):
        if filename is None:
            f = open(path + name + '_' + "%03d.txt" % (batch_i + id_from), "w")
        else:
            f = open(path + filename + '.txt', "w")
        atoms = torch.argmax(one_hot[batch_i], dim=1)
        n_atoms = int(atomsxmol[batch_i])
        lines = []
        real_atoms_num = 0
        for atom_i in range(n_atoms):
            atom = atoms[atom_i]
            atom = dataset_info['atom_decoder'][atom]
            if atom != 'B' or positions[batch_i, atom_i, 0] + positions[batch_i, atom_i, 1] + positions[
                batch_i, atom_i, 2] > 1:
                lines.append("%s %.9f %.9f %.9f\n" % (
                    atom, positions[batch_i, atom_i, 0], positions[batch_i, atom_i, 1], positions[batch_i, atom_i, 2]))
                real_atoms_num += 1
        f.write("%d\n\n" % real_atoms_num)
        for line in lines:
            f.write(line)
        f.close()


def load_molecule_xyz(file, dataset_info):
    with open(file, encoding='utf8') as f:
        n_atoms = int(f.readline())
        one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
        charges = torch.zeros(n_atoms, 1)
        positions = torch.zeros(n_atoms, 3)
        f.readline()
        atoms = f.readlines()
        for i in range(n_atoms):
            atom = atoms[i].split(' ')
            atom_type = atom[0]
            one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
            position = torch.Tensor([float(e) for e in atom[1:]])
            positions[i, :] = position
        return positions, one_hot, charges


def load_xyz_files(path, shuffle=True):
    files = glob.glob(path + "/*.txt")
    if shuffle:
        random.shuffle(files)
    return files


# <----########
### Files ####
##############
def draw_sphere(ax, x, y, z, size, color, alpha):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    xs = size * np.outer(np.cos(u), np.sin(v))
    ys = size * np.outer(np.sin(u), np.sin(v)) * 0.8  # Correct for matplotlib.
    zs = size * np.outer(np.ones(np.size(u)), np.cos(v))
    # for i in range(2):
    #    ax.plot_surface(x+random.randint(-5,5), y+random.randint(-5,5), z+random.randint(-5,5),  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    ax.plot_surface(x + xs, y + ys, z + zs, rstride=2, cstride=2, color=color, linewidth=0,
                    alpha=alpha)
    # # calculate vectors for "vertical" circle
    # a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
    # b = np.array([0, 1, 0])
    # b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (
    #             1 - np.cos(rot))
    # ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle='dashed')
    # horiz_front = np.linspace(0, np.pi, 100)
    # ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
    # vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
    # ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u),
    #         a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle='dashed')
    # ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front),
    #         b[1] * np.cos(vert_front),
    #         a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
    #
    # ax.view_init(elev=elev, azim=0)


def plot_molecule(ax, positions, atom_type, alpha, spheres_3d, hex_bg_color,
                  dataset_info):
    # draw_sphere(ax, 0, 0, 0, 1)
    # draw_sphere(ax, 1, 1, 1, 1)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]
    # Hydrogen, Carbon, Nitrogen, Oxygen, Flourine

    # ax.set_facecolor((1.0, 0.47, 0.42))
    colors_dic = np.array(dataset_info['colors_dic'])
    radius_dic = np.array(dataset_info['radius_dic'])
    area_dic = 1500 * radius_dic ** 2
    # areas_dic = sizes_dic * sizes_dic * 3.1416

    areas = area_dic[atom_type]
    radii = radius_dic[atom_type]
    colors = colors_dic[atom_type]

    if spheres_3d:
        for i, j, k, s, c in zip(x, y, z, radii, colors):
            draw_sphere(ax, i.item(), j.item(), k.item(), 0.7 * s, c, alpha)
    else:
        ax.scatter(x, y, z, s=areas, alpha=0.9 * alpha,
                   c=colors)  # , linewidths=2, edgecolors='#FFFFFF')

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            p1 = np.array([x[i], y[i], z[i]])
            p2 = np.array([x[j], y[j], z[j]])
            dist = np.sqrt(np.sum((p1 - p2) ** 2))
            atom1, atom2 = dataset_info['atom_decoder'][atom_type[i]], \
                dataset_info['atom_decoder'][atom_type[j]]
            s = sorted((atom_type[i], atom_type[j]))
            pair = (dataset_info['atom_decoder'][s[0]],
                    dataset_info['atom_decoder'][s[1]])

            draw_edge_int = bond_analyze.get_bond_order(atom1, atom2, dist)
            line_width = (3 - 2) * 2

            draw_edge = draw_edge_int > 0
            if draw_edge:
                if draw_edge_int == 4:
                    linewidth_factor = 1.5
                else:
                    # linewidth_factor = draw_edge_int  # Prop to number of
                    # edges.
                    linewidth_factor = 1
                ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]],
                        linewidth=line_width * linewidth_factor,
                        c=hex_bg_color, alpha=alpha)


def plot_data3d(positions, atom_type, dataset_info, camera_elev=0, camera_azim=0, save_path=None, spheres_3d=False,
                bg='black', alpha=1., max_v=None):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                  hex_bg_color, dataset_info)
    if max_v is None:
        max_value = positions.abs().max().item()
    else:
        max_value = max_v
    # axis_lim = 3.2
    # axis_lim = min(40000000, max(max_value / 1.5 + 0.3, 3.2))
    axis_lim = max_value
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 320 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_data3d_uncertainty(
        all_positions, all_atom_types, dataset_info, camera_elev=0, camera_azim=0,
        save_path=None, spheres_3d=False, bg='black', alpha=1.):
    black = (0, 0, 0)
    white = (1, 1, 1)
    hex_bg_color = '#FFFFFF' if bg == 'black' else '#666666'

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect('auto')
    ax.view_init(elev=camera_elev, azim=camera_azim)
    if bg == 'black':
        ax.set_facecolor(black)
    else:
        ax.set_facecolor(white)
    # ax.xaxis.pane.set_edgecolor('#D0D0D0')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    ax._axis3don = False

    if bg == 'black':
        ax.w_xaxis.line.set_color("black")
    else:
        ax.w_xaxis.line.set_color("white")

    for i in range(len(all_positions)):
        positions = all_positions[i]
        atom_type = all_atom_types[i]
        plot_molecule(ax, positions, atom_type, alpha, spheres_3d,
                      hex_bg_color, dataset_info)

    max_value = all_positions[0].abs().max().item()

    # axis_lim = 3.2
    axis_lim = min(40, max(max_value + 0.3, 3.2))
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_zlim(-axis_lim, axis_lim)

    dpi = 120 if spheres_3d else 50

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=dpi)

        if spheres_3d:
            img = imageio.imread(save_path)
            img_brighter = np.clip(img * 1.4, 0, 255).astype('uint8')
            imageio.imsave(save_path, img_brighter)
    else:
        plt.show()
    plt.close()


def plot_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    im1 = np.arange(100).reshape((10, 10))
    im2 = im1.T
    im3 = np.flipud(im1)
    im4 = np.fliplr(im2)

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(6, 6),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, im in zip(grid, [im1, im2, im3, im4]):
        # Iterating over the grid returns the Axes.

        ax.imshow(im)
    plt.show()


def visualize(path, max_num=10000, wandb=None, spheres_3d=False):
    files = load_xyz_files(path)[0:max_num]
    for file in files:
        if 'txt' in file:
            dataset_info = anion_data_info if "anion" in file else cation_data_info
            positions, one_hot, charges = load_molecule_xyz(file, dataset_info)
            atom_type = torch.argmax(one_hot, dim=1).numpy()
            dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
            dists = dists[dists > 0]
            # print("Average distance between atoms", dists.mean().item())
            plot_data3d(positions, atom_type, dataset_info=dataset_info, save_path=file[:-4] + '.png',
                        spheres_3d=spheres_3d)

            if wandb is not None:
                path = file[:-4] + '.png'
                # Log image(s)
                im = plt.imread(path)
                wandb.log({'molecule': [wandb.Image(im, caption=path)]})


def visualize_chain(path, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)

    def file_for_anion(f):
        return 'anion' in f

    def file_for_cation(f):
        return 'cation' in f

    anion_files = filter(file_for_anion, files)
    cation_files = filter(file_for_cation, files)
    anion_files = sorted(anion_files)
    cation_files = sorted(cation_files)

    files = sorted(files)
    anion_save_paths = []
    cation_save_paths = []
    anion_max_value = 0
    cation_max_value = 0
    for i in range(len(anion_files)):
        file = anion_files[i]
        dataset_info = anion_data_info
        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=dataset_info)
        anion_max_value = max(anion_max_value, positions.abs().max().item())

    for i in range(len(cation_files)):
        file = cation_files[i]
        dataset_info = cation_data_info
        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=dataset_info)
        cation_max_value = max(cation_max_value, positions.abs().max().item())

    for i in range(len(anion_files)):
        file = anion_files[i]

        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=anion_data_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=anion_data_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0, max_v=anion_max_value)
        anion_save_paths.append(fn)

    for i in range(len(cation_files)):
        file = cation_files[i]

        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=cation_data_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=cation_data_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0, max_v=cation_max_value)
        cation_save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in anion_save_paths]
    dirname = os.path.dirname(anion_save_paths[0])
    anion_gif_path = dirname + '/anion_output.gif'
    print(f'Creating gif with {len(imgs)} images for anions')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(anion_gif_path, imgs, subrectangles=True)

    imgs = [imageio.imread(fn) for fn in cation_save_paths]
    dirname = os.path.dirname(cation_save_paths[0])
    cation_gif_path = dirname + '/cation_output.gif'
    print(f'Creating gif with {len(imgs)} images for cations')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(cation_gif_path, imgs, subrectangles=True)


def ion_visualize_chain(path, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)

    def file_for_ion(f):
        return 'ion' in f

    files = filter(file_for_ion, files)
    files = sorted(files)

    save_paths = []

    max_value = 0

    for i in range(len(files)):
        file = files[i]
        dataset_info = anion_data_info if "anion" in files[0] else cation_data_info
        positions, one_hot, charges = load_molecule_xyz(file, dataset_info=dataset_info)
        max_value = max(max_value, positions.abs().max().item())

    for i in range(len(files)):
        file = files[i]

        positions, one_hot, charges = load_molecule_xyz(
            file, dataset_info=anion_data_info if "anion" in files[0] else cation_data_info)
        atom_type = torch.argmax(one_hot, dim=1).numpy()
        fn = file[:-4] + '.png'
        plot_data3d(positions, atom_type, dataset_info=anion_data_info if "anion" in files[0] else cation_data_info,
                    save_path=fn, spheres_3d=spheres_3d, alpha=1.0, max_v=max_value)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images for ions')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)


def visualize_chain_uncertainty(
        path, dataset_info, wandb=None, spheres_3d=False, mode="chain"):
    files = load_xyz_files(path)
    files = sorted(files)
    save_paths = []

    for i in range(len(files)):
        if i + 2 == len(files):
            break

        file = files[i]
        file2 = files[i + 1]
        file3 = files[i + 2]

        positions, one_hot, _ = load_molecule_xyz(file, dataset_info=dataset_info)
        positions2, one_hot2, _ = load_molecule_xyz(
            file2, dataset_info=dataset_info)
        positions3, one_hot3, _ = load_molecule_xyz(
            file3, dataset_info=dataset_info)

        all_positions = torch.stack([positions, positions2, positions3], dim=0)
        one_hot = torch.stack([one_hot, one_hot2, one_hot3], dim=0)

        all_atom_type = torch.argmax(one_hot, dim=2).numpy()
        fn = file[:-4] + '.png'
        plot_data3d_uncertainty(
            all_positions, all_atom_type, dataset_info=dataset_info,
            save_path=fn, spheres_3d=spheres_3d, alpha=0.5)
        save_paths.append(fn)

    imgs = [imageio.imread(fn) for fn in save_paths]
    dirname = os.path.dirname(save_paths[0])
    gif_path = dirname + '/output.gif'
    print(f'Creating gif with {len(imgs)} images')
    # Add the last frame 10 times so that the final result remains temporally.
    # imgs.extend([imgs[-1]] * 10)
    imageio.mimsave(gif_path, imgs, subrectangles=True)

    if wandb is not None:
        wandb.log({mode: [wandb.Video(gif_path, caption=gif_path)]})


def load_sdf(file, dataset_info):
    mol = Chem.SDMolSupplier(file, removeHs=False)[0]
    n_atoms = mol.GetNumAtoms()
    one_hot = torch.zeros(n_atoms, len(dataset_info['atom_decoder']))
    charges = torch.zeros(n_atoms, 1)
    positions = torch.zeros(n_atoms, 3)
    for i in range(n_atoms):
        atom = mol.GetAtomWithIdx(i)
        atom_type = atom.GetSymbol()
        one_hot[i, dataset_info['atom_encoder'][atom_type]] = 1
        position = torch.Tensor([mol.GetConformer().GetAtomPosition(i)])
        positions[i, :] = position
        charges[i] = atom.GetAtomicNum()
    return positions, one_hot, charges


def visualize_sdf(file, dataset_info=None, spheres_3d=True, camera_elev=90, camera_azim=90, noise=0.,
                  eps=None, eps_oh=None):
    positions, one_hot, charges = load_sdf(file, dataset_info)

    if noise != 0:
        if eps is not None:
            positions = eps * noise + positions * (1 - noise ** 2) ** 0.5
            one_hot += eps_oh * noise + one_hot * (1 - noise ** 2) ** 0.5
        else:
            epsilon_position = torch.randn(positions.shape)
            epsilon_position -= epsilon_position.mean()
            epsilon_one_hot = torch.randn(one_hot.shape)
            positions = epsilon_position * noise + positions * (1 - noise ** 2) ** 0.5
            one_hot += epsilon_one_hot * noise + one_hot * (1 - noise ** 2) ** 0.5

    atom_type = torch.argmax(one_hot, dim=1).numpy()
    plot_data3d(positions, atom_type, dataset_info=dataset_info,
                save_path=(file[:-4] + f'_{camera_elev}_{camera_azim}_{noise}' + '.png'
                           ) if noise != 0 else file[:-4] + f'_{camera_elev}_{camera_azim}' + '.png',
                camera_elev=camera_elev, camera_azim=camera_azim,
                spheres_3d=spheres_3d)


if __name__ == '__main__':
    # plot_grid()
    import U_Chem.dataset as dataset
    from U_Chem.dataset_info import anion_data_info, cation_data_info

    """
    matplotlib.use('macosx')

    task = "visualize_molecules"
    task_dataset = 'geom'

    dataset_info = anion_data_info


    class Args:
        batch_size = 1
        num_workers = 0
        filter_n_atoms = None
        datadir = 'qm9/temp'
        dataset = 'qm9'
        remove_h = False
        include_charges = True


    cfg = Args()

    dataloaders, charge_scale = dataset.retrieve_dataloaders(cfg)

    for i, data in enumerate(dataloaders['train']):
        positions = data['positions'].view(-1, 3)
        positions_centered = positions - positions.mean(dim=0, keepdim=True)
        one_hot = data['one_hot'].view(-1, 5).type(torch.float32)
        atom_type = torch.argmax(one_hot, dim=1).numpy()

        plot_data3d(
            positions_centered, atom_type, dataset_info=dataset_info,
            spheres_3d=True)
    """
    visualize_chain(r'ZDataD_Molecules\ZDA_process_mol\debug_0\epoch_1_0\chain')
