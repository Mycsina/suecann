import time
import cProfile
import pstats
import numpy as np
from src.wann.population import Population, PopConfig


def main():
    cfg = PopConfig(pop_size=300)
    rng = np.random.default_rng(42)
    pop = Population(cfg, seed=42)

    # Initialize some random node/connection structures to make it realistic
    for g in pop.genomes:
        for _ in range(rng.integers(1, 10)):
            src = int(rng.integers(0, 22))
            dst = int(rng.integers(22, 27))
            from src.wann.genome import ConnGene

            g.add_connection(
                ConnGene.make(
                    int(rng.integers(100, 1000)),
                    src,
                    dst,
                    int(rng.choice([-1, 1])),
                    True,
                )
            )

    # First generation
    fitnesses = list(rng.random(size=300))
    pop.tell_fitnesses(fitnesses)

    print("Profiling 10 generations of speciate_and_evolve()...")

    pr = cProfile.Profile()
    pr.enable()

    t0 = time.time()
    for gen in range(10):
        # We need to set fitnesses before calling speciate_and_evolve
        pop.tell_fitnesses(list(rng.random(size=300)))
        pop.speciate_and_evolve()
    t1 = time.time()

    pr.disable()

    print(
        f"Total time for 10 generations: {t1 - t0:.3f}s (Average: {(t1 - t0)/10:.4f}s per gen)"
    )

    stats = pstats.Stats(pr).sort_stats("tottime")
    stats.print_stats(20)


if __name__ == "__main__":
    main()
