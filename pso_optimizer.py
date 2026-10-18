import numpy as np

class Particle:
    def __init__(self, bounds):
        self.position = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        self.velocity = np.array([np.random.uniform(-0.1, 0.1) for _ in bounds])
        self.best_position = np.copy(self.position)
        self.best_score = -float('inf')

class SwarmOptimizerPSO:
    """
    Optimiseur PSO (Particle Swarm Optimization) pour l'optimisation
    des hyperparamètres (Learning Rate, Weight Decay) dans le cadre du Master.
    """
    def __init__(self, num_particles: int, max_iter: int, bounds: list):
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.bounds = bounds
        self.particles = [Particle(bounds) for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_score = -float('inf')

    def optimize(self, fitness_function):
        w = 0.5   # Facteur d'inertie
        c1 = 1.5  # Coefficient cognitif
        c2 = 1.5  # Coefficient social

        print("=================================================================")
        print("   OPTIMISATION PSO (PARTICLE SWARM OPTIMIZATION) — RUN   ")
        print("=================================================================\n")

        for iteration in range(self.max_iter):
            for particle in self.particles:
                score = fitness_function(particle.position)
                
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = np.copy(particle.position)
                    
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = np.copy(particle.position)

            for particle in self.particles:
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive = c1 * r1 * (particle.best_position - particle.position)
                social = c2 * r2 * (self.global_best_position - particle.position)
                particle.velocity = w * particle.velocity + cognitive + social
                particle.position += particle.velocity

                for i in range(len(self.bounds)):
                    particle.position[i] = np.clip(particle.position[i], self.bounds[i][0], self.bounds[i][1])

            print(f"[PSO] Itération {iteration+1:02d}/{self.max_iter:02d} | Meilleurs hyperparamètres : LR={self.global_best_position[0]:.6f}, WD={self.global_best_position[1]:.6f} | Fitness (Accuracy/F1): {self.global_best_score:.4f}")

        print("\n-----------------------------------------------------------------")
        print(f" OPTIMISATION TERMINÉE avec succès !")
        print(f"  - Learning Rate optimal (LR)  : {self.global_best_position[0]:.6f}")
        print(f"  - Weight Decay optimal (WD)   : {self.global_best_position[1]:.6f}")
        print(f"  - Score de performance (F1)   : {self.global_best_score:.4f}")
        print("-----------------------------------------------------------------\n")

        return self.global_best_position, self.global_best_score

# --- Bloc d'exécution principal ---
if __name__ == "__main__":
    # Bornes de recherche : [Learning Rate (1e-4 à 1e-2), Weight Decay (1e-5 à 1e-3)]
    search_bounds = [(0.0001, 0.01), (0.00001, 0.001)]
    
    # Fonction fitness objective de simulation (maximiser le F1-score)
    def dummy_fitness(params):
        lr, wd = params[0], params[1]
        # Simulation mathématique d'une surface de réponse convexe centrée sur une zone optimale
        score = 0.95 - (lr - 0.001)**2 * 100 - (wd - 0.0001)**2 * 1000
        return float(np.clip(score, 0.50, 0.98))

    optimizer = SwarmOptimizerPSO(num_particles=15, max_iter=10, bounds=search_bounds)
    optimizer.optimize(dummy_fitness)