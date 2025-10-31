import random
import matplotlib.pyplot as plt


class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.x = 0
        self.y = 0


class Tree:
    def __init__(self):
        self.root = None

    def add_node(self, key, node=None):
        if node is None:
            node = self.root

        if self.root is None:
            self.root = Node(key)
        else:
            if key <= node.key:
                if node.left is None:
                    node.left = Node(key)
                    node.left.parent = node
                    return
                else:
                    return self.add_node(key, node=node.left)
            else:
                if node.right is None:
                    node.right = Node(key)
                    node.right.parent = node
                    return
                else:
                    return self.add_node(key, node=node.right)


class KnuthVisualization:
    def __init__(self, tree):
        self.tree = tree
        self.positions = {}
        self.counter = 0

    def assign_positions(self, node, depth=0):
        if node is None:
            return
        self.assign_positions(node.right, depth + 1)  # Step down right
        node.x = self.counter
        node.y = -depth
        self.positions[node] = (node.x, node.y)
        self.counter += 1
        self.assign_positions(node.left, depth + 1)  # Step down left

    def draw_tree(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('Binary tree')
        ax.axis('off')

        def draw_edges(node):
            if node is None:
                return
            if node.left:
                x1, y1 = self.positions[node]
                x2, y2 = self.positions[node.left]
                ax.plot([x1, x2], [y1, y2], color='gray')
                draw_edges(node.left)
            if node.right:
                x1, y1 = self.positions[node]
                x2, y2 = self.positions[node.right]
                ax.plot([x1, x2], [y1, y2], color='gray')
                draw_edges(node.right)

        draw_edges(self.tree.root)

        for node, (x, y) in self.positions.items():
            ax.scatter(x, y, color='pink', s=500, edgecolors='black')
            ax.text(x, y, str(node.key), ha='center', va='center', fontsize=10, weight='bold')

        plt.show()


t = Tree()

for value in random.sample(range(-500, 1), 50):
    t.add_node(value)

viz = KnuthVisualization(t)
viz.assign_positions(t.root)
viz.draw_tree()
