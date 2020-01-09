# The following code is adapted from Mozilla DeepSpeech
# at https://github.com/mozilla/DeepSpeech
# mozilla/DeepSpeech is licensed under the Mozilla Public License 2.0


def wer(original, result):
    """
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:

    # original = original.split()
    # result = result.split()

    wer = levenshtein(original, result) / float(len(original))
    if wer > 1.0:
        return 1.0
    else:
        return wer


def wers(originals, results):
    ops = {'insert':0,'delete':0,'replace':0}
    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)

        # ops_list = Levenshtein.editops(originals[i], results[i])
        # for op in ops_list:
        #     ops[op[0]] += 1
        # #找相同部分
        # mb = Levenshtein.matching_blocks(ops_list, originals[i], results[i])
        # same = ''.join([originals[i][x[0]:x[0]+x[2]] for x in mb])
        # print(same)
    return rates, mean / float(count)

import Levenshtein
def wers2(originals, results):
    ops = {'insert':0,'delete':0,'replace':0}
    t_each_tone = {'0':0, '1':0, '2':0, '3':0, '4':0}
    total_each_tone = {'0':0, '1':0, '2':0, '3':0, '4':0}

    count = len(originals)
    try:
        assert count > 0
    except:
        print(originals)
        raise("ERROR assert count>0 - looks like data is missing")
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = wer(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)

        ops_list = Levenshtein.editops(originals[i], results[i])
        for op in ops_list:
            ops[op[0]] += 1
        #找相同部分
        mb = Levenshtein.matching_blocks(ops_list, originals[i], results[i])
        same = ''.join([originals[i][x[0]:x[0]+x[2]] for x in mb])
        for s in same:
            t_each_tone[s] += 1
        for o in originals[i]:
            total_each_tone[o] += 1
    for i in total_each_tone:
        #if total_each_tone[i] != 0:
        t_each_tone[i] = t_each_tone[i] / total_each_tone[i]
        #else:
        #    t_each_tone[i] = 0
    # print(ops)
    return rates, mean / float(count), ops, t_each_tone

# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

import numpy as np

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
    return current[n]
