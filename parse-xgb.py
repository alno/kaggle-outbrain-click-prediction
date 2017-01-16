import pandas as pd
import numpy as np

import seaborn as sns

import re

node_regex = re.compile("(\d+):\[(.*)<(.+)\]\syes=(.*),no=(.*),missing=.*,gain=(.*),cover=(.*)")
leaf_regex = re.compile("(\d+):leaf=(.*),cover=(.*)")


def merge_feature_importances_inplace(to_imp, from_imp):
    for f in from_imp:
        if f in to_imp:
            to_imp[f] += from_imp[f]
        else:
            to_imp[f] = from_imp[f]


class FeatureImportance(object):
    def __init__(self, expected_fscore, expected_gain):
        self.expected_fscore = expected_fscore
        self.expected_gain = expected_gain

    def __add__(self, other):
        return FeatureImportance(
            expected_fscore=self.expected_fscore + other.expected_fscore,
            expected_gain=self.expected_gain + other.expected_gain
        )

    def __repr__(self):
        return "[expected_fscore=%.5f, expected_gain=%.5f]" % (self.expected_fscore, self.expected_gain)


class XgbLeaf(object):
    def __init__(self, index, value, cover):
        self.index = index
        self.value = value
        self.cover = cover

    def collect_feature_importances(self, importances, path_probability):
        pass

    def collect_split_values(self, feature, values):
        pass

    def constrain(self, feature, value):
        return self


class XgbTree(object):
    def __init__(self, index, feature, split_value, gain, cover):
        self.index = index
        self.feature = feature
        self.split_value = split_value
        self.gain = gain
        self.cover = cover

    def collect_feature_importances(self, importances, path_probability):
        importance = FeatureImportance(expected_fscore=path_probability, expected_gain=path_probability * self.gain)

        if self.feature in importances:
            importances[self.feature] += importance
        else:
            importances[self.feature] = importance

        self.left.collect_feature_importances(importances, path_probability=path_probability * self.left.cover / self.cover)
        self.right.collect_feature_importances(importances, path_probability=path_probability * self.right.cover / self.cover)

    def collect_split_values(self, feature, values):
        if feature == self.feature:
            if self.split_value in values:
                values[self.split_value] += 1
            else:
                values[self.split_value] = 1

        self.left.collect_split_values(feature, values)
        self.right.collect_split_values(feature, values)

    def constrain(self, feature, value):
        if feature == self.feature:
            if value < self.split_value:
                return self.left.constrain(feature, value)
            else:
                return self.right.constrain(feature, value)

        tree = XgbTree(index=self.index, feature=self.feature, split_value=self.split_value, gain=self.gain, cover=self.cover)
        tree.left = self.left.constrain(feature, value)
        tree.right = self.right.constrain(feature, value)

        return tree


class XgbModel(object):
    def __init__(self, trees):
        self.trees = trees

    def get_feature_importances(self):
        importances = {}

        for tree in self.trees:
            tree.collect_feature_importances(importances, path_probability=1.0)

        return importances

    def get_split_values(self, feature):
        values = {}

        for tree in self.trees:
            tree.collect_split_values(feature, values)

        return values

    def constrain(self, feature, value):
        return XgbModel([tree.constrain(feature, value) for tree in self.trees])



def parse_node(f):
    line = f.readline().strip()

    if 'leaf' in line:
        m = leaf_regex.match(line)

        return XgbLeaf(
            index=int(m.group(1)),
            value=float(m.group(2)),
            cover=float(m.group(3))
        )
    else:
        m = node_regex.match(line)

        tree = XgbTree(
            index=int(m.group(1)),
            feature=m.group(2),
            split_value=float(m.group(3)),
            gain=float(m.group(6)),
            cover=float(m.group(7))
        )

        left_index = int(m.group(4))
        right_index = int(m.group(5))

        first = parse_node(f)
        second = parse_node(f)

        if first.index == left_index and second.index == right_index:
            tree.left = first
            tree.right = second
        elif first.index == right_index and second.index == left_index:
            tree.left = second
            tree.right = first
        else:
            raise RuntimeError("Mismatching tree indices")

        return tree


def parse_model_dump(file_name):
    with open(file_name) as f:
        trees = []

        while True:
            line = f.readline().strip()

            if not line:
                break
            elif 'booster' in line:
                trees.append(parse_node(f))
            else:
                raise RuntimeError("Can't parse line: '%s'" % line)

        return XgbModel(trees)


model = parse_model_dump('xg.v10.dump')

time_split_values = model.get_split_values('time').keys()

min_time_value = min(time_split_values) - 1.0
max_time_value = max(time_split_values) + 1.0

time_values = np.linspace(min_time_value, max_time_value, 200)

gain_records = []
fscore_records = []

for time_value in time_values:
    imps = model.constrain('time', time_value).get_feature_importances()

    gain_records.append({f: imps[f].expected_gain for f in imps})
    fscore_records.append({f: imps[f].expected_fscore for f in imps})


df_gain = pd.DataFrame.from_records(gain_records, index=map(int, time_values))
df_gain.index.rename('time', inplace=True)
df_gain.to_csv('xg.v10.gain.csv')

df_fscore = pd.DataFrame.from_records(fscore_records, index=map(int, time_values))
df_fscore.index.rename('time', inplace=True)
df_fscore.to_csv('xg.v10.fscore.csv')
