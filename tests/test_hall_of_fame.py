import os
import tempfile
import pytest
from src.wann.genome import Genome, ConnGene
from src.oracle.hall_of_fame import HallOfFame


def test_hall_of_fame_save_load():
    # 1. Create a couple of genomes with connections
    g1 = Genome.initial()
    g1.add_connection(ConnGene.make(100, 0, 19, sign=1, enabled=True))
    g1.add_connection(ConnGene.make(101, 1, 20, sign=-1, enabled=True))

    g2 = Genome.initial()
    g2.add_connection(ConnGene.make(200, 2, 21, sign=-1, enabled=True))

    # 2. Add to Hall of Fame
    hof = HallOfFame(max_size=5)
    hof.add(g1, fitness=10.0, generation=1)
    hof.add(g2, fitness=5.0, generation=2)

    # 3. Save to temp file and load back
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        temp_path = f.name

    try:
        hof.save(temp_path)
        loaded = HallOfFame.load(temp_path)

        assert len(loaded) == 2

        # Verify first entry (g1, fitness 10)
        e0 = loaded.entries[0]
        assert e0.fitness == 10.0
        assert e0.generation == 1
        assert len(e0.genome.conn_genes) == 2
        assert 100 in e0.genome.conn_genes
        assert 101 in e0.genome.conn_genes
        assert e0.genome.conn_genes[100].src == 0
        assert e0.genome.conn_genes[100].dst == 19
        assert e0.genome.conn_genes[100].sign == 1
        assert e0.genome.conn_genes[100].enabled is True

        # Verify second entry (g2, fitness 5)
        e1 = loaded.entries[1]
        assert e1.fitness == 5.0
        assert e1.generation == 2
        assert len(e1.genome.conn_genes) == 1
        assert 200 in e1.genome.conn_genes
        assert e1.genome.conn_genes[200].src == 2
        assert e1.genome.conn_genes[200].dst == 21
        assert e1.genome.conn_genes[200].sign == -1
        assert e1.genome.conn_genes[200].enabled is True

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
