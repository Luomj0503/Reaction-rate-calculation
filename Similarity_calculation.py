import numpy as np
from numpy import log, e
from math import sqrt
import pandas as pd
from tqdm import tqdm

class Similarity:
    def __init__(self):
        pass

    def __coefs(self, vetor1, vetor2):
        A = np.array(vetor1).astype(int)
        B = np.array(vetor2).astype(int)

        AnB = A & B  # intersection
        onlyA = np.array(B) < np.array(A)  # A is a subset of B
        onlyB = np.array(A) < np.array(B)  # B is a subset of A
        AuB_0s = A | B  # Uniao (for count de remain zeros)

        return AnB, onlyA, onlyB, np.count_nonzero(AuB_0s == 0)

    def tanimoto_similarity(self, vetor1, vetor2):
        """
        Structural similarity calculation based on tanimoto index.
        T(A,B) = (A ^ B)/(A + B - A^B)
        """
        AnB, onlyA, onlyB, AuB_0s = Similarity.__coefs(self, vetor1=vetor1, vetor2=vetor2)
        return AnB.sum() / (onlyA.sum() + onlyB.sum() + AnB.sum())


class ApplicabilityDomain:
    def __init__(self, verbose=False):
        self.__sims = Similarity()
        self.__verbose = verbose
        self.similarities_table_ = None

    def analyze_similarity(self, base_test, base_train):
        print(base_test, base_train)
        get_tests_similarities = []
        similarities = []
        sim = []
        # get dictionary of all data tests similarities
        for n,i_train in enumerate(base_train):
            get_tests_similarities.append(self.__sims.tanimoto_similarity(base_test, i_train))
            similarities = np.array(get_tests_similarities)
            get_tests_similarities = []
            sim.append(similarities)
        sim = np.array(sim)
        self.similarities_table_ = pd.DataFrame(sim)
        analyze = pd.concat([self.similarities_table_.mean(),
                             self.similarities_table_.median(),
                             self.similarities_table_.std(),
                             self.similarities_table_.max(),
                             self.similarities_table_.min()],
                            axis=1)
        analyze.columns = ['Mean', 'Median', 'Std', 'Max', 'Min']

        return analyze
