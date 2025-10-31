import os
import sys
import subprocess
import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import networkx as nx


def is_hidden_dir(d):
    if sys.platform.startswith('win'):
        try:
            p = subprocess.check_output(['attrib', d], shell=True)
            p = p.decode('cp1251', errors='replace')  #
            return 'H' in p[:12]
        except Exception:
            return False
    else:
        return os.path.basename(d).startswith('.')


def get_tree(tree, G=nx.Graph(), itr=0, max_itr=900):
    point = tree.pop(0)
    itr += 1
    sub_tree = [os.path.join(point, x) for x in os.listdir(point)
                if os.path.isdir(os.path.join(point, x)) and not is_hidden_dir(os.path.join(point, x))]

    if sub_tree:
        tree.extend(sub_tree)
        G.add_edges_from((point, b) for b in sub_tree)

    for f in os.listdir(point):
        f_path = os.path.join(point, f)
        if os.path.isfile(f_path):
            G.add_edge(point, f_path)

    if tree and itr <= max_itr:
        return get_tree(tree, G, itr)
    else:
        return G


def get_creation_time(path):
    try:
        ts = os.path.getctime(path)
        return datetime.datetime.fromtimestamp(ts)
    except Exception:
        return datetime.datetime(2000, 1, 1)


def visualize_tree(root_dir):
    G = get_tree(tree=[root_dir])

    node_dates = {n: get_creation_time(n) for n in G.nodes if os.path.isdir(n)}

    if node_dates:
        min_date = min(node_dates.values())
        max_date = max(node_dates.values())
        date_range = (max_date - min_date).total_seconds()
    else:
        date_range = 1

    norm = mcolors.Normalize(vmin=0, vmax=date_range)
    cmap = cm.viridis

    node_colors = []
    for n in G.nodes:
        if os.path.isdir(n):
            delta = (get_creation_time(n) - min_date).total_seconds()
            color = cmap(norm(delta))
        else:
            color = mcolors.to_rgba("#cd5c5c", alpha=0.9)
        node_colors.append(color)

    options = {
        'node_color': node_colors,
        'node_size': 40,
        'width': 0.4,
        'with_labels': False,
        'alpha': 0.8
    }

    plt.figure(figsize=(12, 10))
    plt.title('Directories and files\n', fontsize=12)

    pos = nx.spring_layout(G, seed=42)
    ax = plt.gca()
    nx.draw(G, pos, ax=ax, **options)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Folder creation date from old to new', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root_dir = r'D:\prog\pythonProjects\mathStatistics'  # r'D:\prog'
    visualize_tree(root_dir)
