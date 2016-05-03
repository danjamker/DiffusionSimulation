from __future__ import division

import operator
import scipy.stats


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
        self.avrageUserExposure = 0
        self.sequence = []
        self.time_sequence = []
        self.ActivateionExposureArray = []
        self.UserExposureArray = []
        self.inffectedCommunities = 0
        self.window = 100
        self.avrage_time_set = 0
        self.cv_avrage_time_set = 0
        self.diamiter = 0

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
                print self.userUsageEntorpy
                print [self.activatedUsersPerCommunity[c] for c in self.Communities]

                self.ActivateionExposure = sum([self.G.node[ns]['activated'] for ns in self.G.neighbors(n) if
                                                self.G.node[ns]['activated'] > 0])

                self.ActivateionExposureArray.append(self.ActivateionExposure)
                self.avrageActivateionExposure = sum(self.ActivateionExposureArray) / float(
                    len(self.ActivateionExposureArray))

                self.UserExposure = len([self.G.node[ns]['activated'] for ns in self.G.neighbors(n) if
                                         self.G.node[ns]['activated'] > 0])

                self.UserExposureArray.append(self.UserExposure)
                self.avrageUserExposure = sum(self.UserExposureArray) / float(len(self.UserExposureArray))

                self.inffectedCommunities = len({k: v for k, v in self.activatedUsersPerCommunity.items() if v > 0})

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
