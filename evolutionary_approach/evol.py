import math
import numpy as np
import gym
from collections import Counter
import multiprocessing as mp
import os


class evolution:
    @staticmethod
    def most_common(lst):
        data = Counter(lst)
        return max(lst, key=data.get)

    def __init__(self, action_space, population_number, length, mutation_chance=0.2):
        self.action_space = action_space
        self.population = [list(np.random.randint(0, self.action_space, size=length)) for _ in range(population_number)]
        self.length = length
        self.population_number = population_number
        self.env = gym.make("CarRacing-v2", continuous=False)  # , render_mode ='human')
        self.scores = np.zeros(self.population_number)
        self.done = False
        self.mutation_chance = mutation_chance
        self.num_processes = os.cpu_count() - 2

    def load(self, name):
        self.population = [list(np.load(name)) for _ in range(self.population_number)]
        self.population_number = len(self.population)
        self.length = np.shape(self.population)[1]
        self.scores = [1 for _ in range(self.population_number)]
        self.next_gen()

    def eval_population(self):
        for i in range(self.population_number):
            reward = 0
            self.env.reset(seed=27)
            for j in range(self.length):
                action = self.population[i][j]
                state_next, r, _, _, done = self.env.step(action)
                reward += r
                if done:
                    self.done = True
                    break
            self.scores[i] = reward

    # def eval_population(self):
    #    with mp.Pool(self.num_processes) as pool:
    #        self.scores = np.array(pool.map(self.eval_individual, self.population))
    #
    def eval_individual(self, individual, queue, num):
        rew = []
        for i in range(len(individual)):
            reward = 0
            self.env.reset(seed=27)
            for j in range(self.length):
                action = individual[i][j]
                state_next, r, _, _, done = self.env.step(action)
                if action == 3:
                    r += 0.2
                reward += r
                if done:
                    break
            rew.append(reward)
        queue.put({num: (rew, individual)})

    def mp_eval(self):
        queue = mp.Queue()
        chunks = int(math.ceil(self.population_number / self.num_processes))
        procs = []
        results = {}
        for i in range(self.num_processes):
            proc = mp.Process(target=self.eval_individual,
                              args=(self.population[i * chunks:(i + 1) * chunks], queue, i))
            procs.append(proc)
            proc.start()
        for i in range(self.num_processes):
            results.update(queue.get())
        for i in procs:
            i.join()
        new_pop = []
        score = []
        for i in range(self.num_processes):
            for l in range(len(results[i][0])):
                new_pop.append(results[i][1][l])
                score.append(results[i][0][l])
        self.population = new_pop
        self.scores = score

    def next_gen(self):
        for i in range(len(self.scores)):
            self.scores[i] = np.max([self.scores[i], 0])
        self.scores = self.scores / np.sum(self.scores)
        new_pop = []
        arr = np.linspace(0, self.population_number - 1, self.population_number)
        new_pop.append(self.population[np.argmax(self.scores)].copy())
        most = [self.most_common(np.array(new_pop.copy())[:, i]) for i in range(self.length)]
        new_pop.append(most.copy())
        for i in range(self.population_number - 2):
            index = int(np.random.choice(arr, p=self.scores))
            new_pop.append(self.population[index].copy())
        for i in range(2, self.population_number):
            for j in range(self.length-20, self.length):
                r = np.random.rand()
                if r < self.mutation_chance/2:
                    new_pop[i][j] = most[j]
                elif 1 - self.mutation_chance < r:
                    new_pop[i][j] = np.random.randint(0, self.action_space)
        self.population = new_pop

    def append_n(self, n=10):
        if not self.done:
            for _ in range(n):
                for i in range(self.population_number):
                    self.population[i].append(np.random.randint(0, self.action_space))
                self.length += 1
