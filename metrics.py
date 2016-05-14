from __future__ import division

import logging
import operator
import scipy.stats
import numpy as np
class metric:
    def __init__(self, G):
        self.G = G
        self.Communities = list(set([attrdict['community'] for n, attrdict in self.G.node.items()]))
        self.numberActivatedUsers = 0
        self.numberOfActivations = 0
        self.usagedominance = 0
        self.activationsPerCommunity = dict((k, 0) for k in self.Communities)
        self.activatedUsersPerCommunity = dict((k, 0) for k in self.Communities)
        self.userUsageDominance = 0
        self.usageEntorpy = 0
        self.userUsageEntorpy = 0
        self.ActivateionExposure = 0
        self.UserExposure = 0
        self.avrageActivateionExposure = 0
        self.cvActivateionExposure = 0
        self.avrageUserExposure = 0
        self.cvUserExposure = 0
        self.sequence = []
        self.ActivateionExposureArray = []
        self.UserExposureArray = []
        self.inffectedCommunities = 0
        self.window = 100

        #Time
        self.avrage_time_set = 0
        self.cv_avrage_time_set = 0
        self.time_dif_sequence = []
        self.current_time = None

        #Diamiter
        self.diamiter = 0

        #Surface
        self.surface_step = []
        self.surface_mean = 0
        self.surface_cv = 0
        self.surface = 0
        self.surface_set = set()

    def add(self, n, step_time=None):
        if n is not None:
            if self.G.has_node(n):

                node = self.G.node[n]
                self.sequence.append(node)

                if node["activated"] == 1:
                    self.numberActivatedUsers += 1
                    self.activatedUsersPerCommunity[node["community"]] += 1

                self.numberOfActivations += 1
                self.activationsPerCommunity[node["community"]] += 1

                #Compute time diffrence if provided
                if step_time != None:
                    if self.current_time == None:
                        self.current_time = step_time
                    else:
                        self.time_dif_sequence.append( self.self.current_time - step_time)
                        self.current_time = step_time
                        self.avrage_time_set = np.mean(self.time_dif_sequence)
                        self.cv_avrage_time_set = np.std(self.time_dif_sequence)/self.avrage_time_set


                dominatcommunity = max(self.activationsPerCommunity.iteritems(), key=operator.itemgetter(1))[0]

                try:
                    self.usagedominance = self.activationsPerCommunity[dominatcommunity] / self.numberOfActivations
                except:
                    self.usagedominance = 0

                dominatcommunity = max(self.activatedUsersPerCommunity.iteritems(), key=operator.itemgetter(1))[0]
                try:
                    self.userUsageDominance = self.activatedUsersPerCommunity[
                                                  dominatcommunity] / self.numberActivatedUsers
                except:
                    self.userUsageDominance = 0

                # Shouldnt this be a list of number of the number of times a users has used an activation
                self.usageEntorpy = scipy.stats.entropy(
                    [self.activationsPerCommunity[c] for c in self.Communities])

                self.userUsageEntorpy = scipy.stats.entropy(
                    [self.activatedUsersPerCommunity[c] for c in self.Communities])

                #Compute Activations
                exposures = [self.G.node[ns]['activated'] for ns in self.G.neighbors(n) if
                             self.G.node[ns]['activated'] > 0]
                self.ActivateionExposure = sum(exposures)
                self.ActivateionExposureArray.append(self.ActivateionExposure)
                self.avrageActivateionExposure = np.mean(self.ActivateionExposureArray)
                self.cvActivateionExposure = np.std(self.ActivateionExposureArray)/self.avrageActivateionExposure

                #compute user exposure
                self.UserExposure = len(exposures)
                self.UserExposureArray.append(self.UserExposure)
                self.avrageUserExposure = np.mean(self.UserExposureArray)
                self.cvUserExposure = np.stf(self.UserExposureArray)/self.avrageActivateionExposure

                self.inffectedCommunities = len({k: v for k, v in self.activatedUsersPerCommunity.items() if v > 0})

                #Compute values for steps
                self.surface_set |= [x for x in self.G.neighbors(node) if self.G[node][x]['weight'] == 0]
                self.surface = len(self.surface_set)
                self.surface.remove(n)
                self.surface_step.append(len(self.surface_set))
                self.surface_mean = np.mean(self.surface_step)
                self.surface_cv =  np.std(self.surface_step)/self.surface_step

    def asMap(self):
        return {"numberActivatedUsers": self.numberActivatedUsers,
                "numberOfActivations": self.numberOfActivations,
                "usagedominance": self.usagedominance,
                "userusagedominance": self.userUsageDominance,
                "usageEntorpy": self.usageEntorpy,
                "userUsageEntorpy": self.userUsageEntorpy,
                "ActivateionExposure": self.ActivateionExposure,
                "UserExposure": self.UserExposure,
                "avrageUserExposure": self.avrageUserExposure,
                "avrageActivateionExposure": self.avrageActivateionExposure,
                "inffectedCommunities": self.inffectedCommunities,
                "cv_avrage_time_set": self.cv_avrage_time_set,
                "avrage_time_set": self.avrage_time_set}
