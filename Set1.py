# Tsevrainhs Iwannhs 2562
# Dionusiadhs Iwannhs 2965
# First series of exercises, Assignment 1:
# Erwthsh 1: A. 3.
# Erwthsh 1: B.
import random


def reservoir_sampling():  # 1 sample out of n
    file_path = 'vocab.nytimes.txt'
    with open(file_path) as fp:
        n = 0
        for cnt, line in enumerate(fp):
            n += 1
            if random.random() < 1 / n:
                selected = line
        print(selected)


def weighted_reservoir_sampling(file):  # assignment 1 subquery B.
    file_path = file
    with open(file_path) as fp:
        total_weight = 0
        for cnt, line in enumerate(fp):
            weight = random.random()  # assign random weight (0~1) to each line
            if weight == 0:
                continue
            total_weight += weight
            if random.random() < weight / total_weight:
                selected = line
        print(f'randomly selected 1 weighted item: \n{selected}')


def sample(file, k):  # assignment 1 subquery A. 3.
    with open(file) as fp:
        i = 0
        reservoir = [0] * k  # static memory
        for cnt, line in enumerate(fp):
            if i < k:  # assign first k items with probability 1
                reservoir[i] = line
            else:
                j = random.randrange(i + 1)
                if j < k:
                    reservoir[j] = line
            i += 1
        # remove new line characters from strings if necessary
        res = []
        for sub in reservoir:
            res.append(sub.replace("\n", ""))
        print(f'{k} random selected items: \n{res}')


if __name__ == '__main__':
    file_path = 'vocab.nytimes.txt'
    sample(file_path, 3)
    # weighted_reservoir_sampling(file_path)
