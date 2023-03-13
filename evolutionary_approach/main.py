from evol import evolution
import numpy as np
import gc

if __name__ == '__main__':
    p = evolution(action_space=5, population_number=300, length=110)
    p.population[0] = [3 for _ in range(p.length)]
    p.load('population.npy')
    episode_number = 0
    record_old = -np.inf
    nxt = 10
    while True:
        episode_number += 1
        p.mp_eval()
        sc = np.max(p.scores)
        print(f'Episode {episode_number} done with maximum score of {sc:.2f} and {p.length} moves and old record {record_old}')
        np.save('population.npy', p.population[np.argmax(p.scores)])
        p.next_gen()
        if episode_number > nxt and record_old + 3 < sc:
            p.append_n()
            record_old = sc
            nxt = episode_number + 10
        gc.collect()
