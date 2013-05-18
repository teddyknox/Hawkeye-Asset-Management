from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Initializators, Mutators
from ham import SVM

# Find negative element
ham = SVM()
def eval_func(genome):
   return ham.gen_alg_eval(*tuple(genome))

def run_main():
   # Genome instance
   genome = G1DList.G1DList(3)
   genome.setParams(rangemin=0.0, rangemax=10.0)

   # Change the initializator to Real values
   genome.initializator.set(Initializators.G1DListInitializatorReal)

   # Change the mutator to Gaussian Mutator
   genome.mutator.set(Mutators.G1DListMutatorRealGaussian)

   # The evaluator function (objective function)
   genome.evaluator.set(eval_func)

   # Genetic Algorithm Instance
   ga = GSimpleGA.GSimpleGA(genome)
   ga.selector.set(Selectors.GRouletteWheel)
   ga.setGenerations(10)

   # Do the evolution
   ga.evolve(freq_stats=1)

   # Best individual
   print ga.bestIndividual()

if __name__ == "__main__":
   run_main()