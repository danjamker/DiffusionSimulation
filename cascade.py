from __future__ import division

import random
from abc import abstractmethod
from copy import deepcopy

import networkx as nx
import numpy as np
import pandas as pd


class cascade:
    def __init__(self, G, itterations=10000):
        self.G = deepcopy(G)
        self.cascase_id = 1
        self.step = 1
        self.d = {}
        self.activated = ""
        self.numberOfNodes = len(self.G.nodes())
        self.numberactivated = 0
        self.n = set([n for n, attrdict in self.G.node.items() if attrdict['activated'] == 0])
        self.a = set([n for n, attrdict in self.G.node.items() if attrdict['activated'] > 0])
        self.iterations = itterations
        self.step_time = None

    def __iter__(self):
        return self

    def decision(self, probability):
        '''
        Returns a True/False dissision bases on a random distribution and a probability threshold.
        :param probability: Probability threshold
        :type probability: int
        :return: True/False
        :rtype: bool
        '''
        return random.random() < probability

    @abstractmethod
    def next(self):
        pass

    def getInfectedNode(self):
        """

        :return:
        :rtype:
        """
        return self.activated

    def getDeepGraph(self):
        return deepcopy(self.G)

    def getGraph(self):
        return self.G

    def getStep(self):
        '''
        Returns the current iteration the cascade is in
        :return: step number
        :rtype: int
        '''
        return self.step

    def getStepTime(self):
        return self.step_time


class randomModel(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            activate = random.choice([n for n, attrdict in self.G.node.items() if attrdict['activated'] == 0])
            self.activated = activate
            if activate != None:
                nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
            self.d[self.step] = len([n for n, attrdict in self.G.node.items() if attrdict['activated'] > 0]) / len(
                self.G.nodes())
            self.step += 1
        else:
            raise StopIteration()


class randomActive(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            if self.decision(0.85):
                activate = random.choice(self.G.nodes())
                self.activated = activate
                if activate != None:
                    nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
                    if self.G.node[activate]['activated'] == 1:
                        self.n.discard(activate)
                        self.a.add(activate)

            else:
                self.cascase_id += 1
                seed = random.sample(self.n, 1)[0]
                self.n.discard(seed)
                self.a.add(seed)
                self.activated = seed
                nx.set_node_attributes(self.G, 'activated', {seed: 1})

            self.step += 1
        else:
            raise StopIteration()


class CascadeNabours(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            if self.decision(0.85):
                node = random.sample(self.a, 1)[0]
                l = self.G.neighbors(node)
                if len(l) > 0:
                    activate = random.choice(l)
                else:
                    activate = None

                self.activated = activate
                if activate != None:
                    nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
                    self.n.discard(activate)
                    self.a.add(activate)
            else:
                self.cascase_id += 1
                seed = random.sample(self.n, 1)[0]
                self.activated = seed
                self.n.discard(seed)
                self.a.add(seed)
                nx.set_node_attributes(self.G, 'activated', {seed: 1})

            self.step += 1
        else:
            raise StopIteration()


class CascadeNaboursWeight(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            if self.decision(0.85):
                node = random.sample(self.a, 1)[0]
                l = self.G.neighbors(node)
                if len(l) > 0:
                    li = [self.G[node][x]['weight'] for x in self.G.neighbors(node)]
                    nn = self.G.neighbors(node)
                    activate = np.random.choice(nn, 1, li)[0]
                else:
                    activate = None

                self.activated = activate
                if activate != None:
                    nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
                    self.n.discard(activate)
                    self.a.add(activate)
            else:
                self.cascase_id += 1
                seed = random.sample(self.n, 1)[0]
                self.activated = seed
                self.n.discard(seed)
                self.a.add(seed)
                nx.set_node_attributes(self.G, 'activated', {seed: 1})

            self.step += 1
        else:
            raise StopIteration()


class NodeWithHighestActiveNabours(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            if self.decision(0.85):
                node = random.sample(self.a, 1)[0]
                activate = self.select(self.G, node)
                self.activated = activate
                if activate != None:
                    self.n.discard(activate)
                    self.a.add(activate)
                    nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
            else:
                self.cascase_id += 1
                seed = random.sample(self.n, 1)[0]
                self.activated = seed
                self.n.discard(seed)
                self.a.add(seed)
                nx.set_node_attributes(self.G, 'activated', {seed: 1})

            self.step += 1
        else:
            raise StopIteration()

    def select(self, G, node):
        t = None
        tc = 0
        for s in G.neighbors(node):
            tmp = G.neighbors(s)
            c = len([n for n in tmp if G.node[n]['activated'] > 0]) / len(tmp)
            if t == None:
                if c > 0:
                    t = s
                    tc = c
            elif c > tc:
                t = s
                tc = c
        return t


class NodeInSameCommunity(cascade):
    def next(self):
        if self.step < self.iterations and len(self.n) > 0:
            if self.decision(0.85):
                node = random.sample(self.a, 1)[0]
                activate = self.select(self.G, node)
                self.activated = activate
                if activate != None:
                    self.n.discard(activate)
                    self.a.add(activate)
                    nx.set_node_attributes(self.G, 'activated', {activate: self.G.node[activate]['activated'] + 1})
            else:
                self.cascase_id += 1
                seed = random.sample(self.n, 1)[0]
                self.activated = seed
                nx.set_node_attributes(self.G, 'activated', {seed: 1})
                self.n.discard(seed)
                self.a.add(seed)

            self.step += 1
        else:
            raise StopIteration()

    def select(self, G, node):
        c = [n for n in G.neighbors(node) if G.node[n]['community'] == G.node[node]["community"]]
        if len(c) > 0:
            return random.choice(c)
        else:
            return None


class actualCascade(cascade):
    def __init__(self, file, G):
        # self.G = deepcopy(G)
        self.G = G
        self.f = file
        self.cascase_id = 1
        self.step = 1
        self.d = {}
        self.activated = ""
        dtf = pd.read_csv(file, index_col=False, header=None, sep="\t", engine="python",
                          compression=None, names=["word", "node", "time"]).drop_duplicates(subset=["time"],
                                                                                            keep='last')
        dtf['time'] = pd.to_datetime(dtf['time'])
        # Filters out users that are not in the network
        dftt = dtf[dtf["node"].isin(self.G.nodes())]
        self.df = dftt.set_index(pd.DatetimeIndex(dftt["time"])).sort_index();
        self.dfi = self.df.iterrows();

        # self.name_to_id = dict((d["name"], n) for n, d in self.G.nodes_iter(data=True))
        self.name_to_id = dict((n, n) for n, d in self.G.nodes_iter(data=True))

    def next(self):
        try:
            activate = next(self.dfi)
            self.activated_name = activate[1]["node"]
            self.step_time = activate[1]["time"]
            self.activated = self.name_to_id[self.activated_name]
            if self.G.has_node(self.name_to_id[self.activated]):
                nx.set_node_attributes(self.G, 'activated',
                                       {self.activated: self.G.node[self.activated]['activated'] + 1})
            else:
                self.activated = None

            self.step += 1
            self.step_time = activate[1]["time"]

        except EOFError:
            raise StopIteration()

        except IndexError:
            raise StopIteration()

        except StopIteration:
            raise StopIteration()
        except KeyError:
            pass


class actualCascadeDF(cascade):
    def __init__(self, JSON, G):
        # self.G = deepcopy(G)
        self.G = G
        self.f = file
        self.cascase_id = 1
        self.step = 1
        self.d = {}
        self.activated = ""
        dtf = pd.read_json(JSON)
        self.df = dtf.sort_index();
        self.dfi = self.df.iterrows();
        self.step_time = None

        # self.name_to_id = dict((d["name"], n) for n, d in self.G.nodes_iter(data=True))
        self.name_to_id = dict((n, n) for n, d in self.G.nodes_iter(data=True))

    def next(self):
        try:
            activate = next(self.dfi)
            self.activated_name = activate[1]['id']
            self.activated = self.name_to_id[self.activated_name]
            if self.G.has_node(self.name_to_id[self.activated]):
                nx.set_node_attributes(self.G, 'activated',
                                       {self.activated: self.G.node[self.activated]['activated'] + 1})
            else:
                self.activated = None

            self.step += 1

        except EOFError:
            raise StopIteration()

        except IndexError:
            raise StopIteration()

        except StopIteration:
            raise StopIteration()
        except KeyError:
            pass
