import sys

import Tools
import networkx as nx

def main(f, t):
    G = nx.read_gpickle(f)
    with open(t, 'w+') as file_:
        for file in G.nodes():
            print(file)
            try:
                file_.write(file + '\n')
            except Exception as e:
                print("Error")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
