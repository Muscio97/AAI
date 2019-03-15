from functools import reduce
from random import shuffle, random, randint


class EA:
    # init population
    def __init__(self, n):
        self.population = self.gen_population(n)

    # Generate random individual of type list of card numbers
    def gen_individual(self):
        cards = [i for i in range(1, 11)]
        shuffle(cards)
        return cards

    # Generate population of individuals of type list of card numbers
    def gen_population(self, n):
        return [self.gen_individual() for i in range(n)]

    # Test individual of its fitness
    def get_fitness(self, individual, n=5):
        pile1 = individual[:n]
        pile2 = individual[n:]

        result1 = sum(pile1)
        result2 = reduce(lambda x, y: x * y, pile2)

        fit1 = abs(36 - result1)
        fit2 = abs(360 - result2)

        return fit1 + fit2

    # Generate children based in parents
    def make_child(self, mother, father):
        gnome_size = len(mother)
        gnome_position = []
        while int(gnome_size / 2) > len(gnome_position):
            r = randint(0, 9)
            if r not in gnome_position:
                gnome_position.append(r)

        child1 = [0 for i in range(gnome_size)]
        child2 = [0 for i in range(gnome_size)]

        motherTemp = mother.copy()
        fatherTemp = father.copy()

        for i in range(int(gnome_size / 2)):
            child1[gnome_position[i]] = mother[gnome_position[i]]
            child2[gnome_position[i]] = father[gnome_position[i]]
            motherTemp.remove(father[gnome_position[i]])
            fatherTemp.remove(mother[gnome_position[i]])

        for i in range(int(gnome_size)):
            if child1[i] == 0:
                child1[i] = fatherTemp.pop()
            if child2[i] == 0:
                child2[i] = motherTemp.pop()

        return [child1, child2]

    # Grade the population
    def get_population_fitness(self):
        return [(self.get_fitness(x), x) for x in self.population]

    # Mutates random children with random crossover
    def mutate(self, children, mutation_rate):
        number_of_mutations = int(len(children) * mutation_rate)
        for _i in range(number_of_mutations):
            child = children[randint(0, len(children) - 1)]
            r1 = randint(0, len(child) - 1)
            r2 = randint(0, len(child) - 1)
            while r1 != r2:
                r2 = randint(0, len(child) - 1)

            temp = child[r1]
            child[r1] = child[r2]
            child[r2] = temp

    # Make next generation
    def next_gen(self, retain, mutation_rate=0.6, random_select=0.6):
        graded = self.get_population_fitness()
        graded = [x[1] for x in sorted(graded)]
        retain_length = int(len(graded) * retain)
        parents = graded[:retain_length]

        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)

        desired_length = len(self.population) - len(parents)
        children = []
        while len(children) < desired_length:
            male = randint(0, len(parents) - 1)
            female = randint(0, len(parents) - 1)
            if male != female:
                children += self.make_child(parents[female], parents[male])

        if len(children) != desired_length:
            children.pop()

        self.mutate(children, mutation_rate)

        self.population = parents
        self.population += children

    # Get one of the best in individual of current generation
    def get_current_best(self):
        graded = self.get_population_fitness()
        graded.sort(key=lambda x: x[0])
        return graded[0]


if __name__ == "__main__":
    oldp = 0

    # Pretty print the sum of population fitness, difference between current and last
    # and best individual of current population
    def print_fitness(e):
        global oldp
        p = [p[0] for p in e.get_population_fitness()]
        print(f"Total: {sum(p)}, Dif: {abs(oldp - sum(p))}, Best: {e.get_current_best()}")
        oldp = sum(p)


    ea = EA(100)
    for i in range(100):
        ea.next_gen(0.8, 0.8)
        print_fitness(ea)
