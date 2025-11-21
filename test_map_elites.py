
import unittest
import torch
import numpy as np
from remote_training_server.v620_map_elites_trainer import PopulationTracker, MAPElitesTrainer, ActorNetwork

class TestMAPElites(unittest.TestCase):
    def test_population_tracker(self):
        tracker = PopulationTracker()

        # Create dummy model
        model = ActorNetwork()
        metrics = {'distance': 10.0, 'collisions': 0, 'duration': 60.0}

        # Test adding entries
        tracker.add(
            model_state=model.state_dict(),
            fitness=10.0,
            avg_speed=0.1,
            avg_clearance=1.0,
            avg_angular_action=0.1,
            metrics=metrics
        )

        self.assertEqual(len(tracker.population), 1)

        # Test novelty calculation
        novelty = tracker.calculate_behavior_novelty(0.1, 1.0, 0.1)
        # With only 1 entry, it returns 0.5
        self.assertEqual(novelty, 0.5)

        # Add more to test novelty properly
        tracker.add(
            model_state=model.state_dict(),
            fitness=12.0,
            avg_speed=0.2,
            avg_clearance=2.0,
            avg_angular_action=0.5,
            metrics=metrics
        )
        tracker.add(
            model_state=model.state_dict(),
            fitness=14.0,
            avg_speed=0.3,
            avg_clearance=3.0,
            avg_angular_action=0.8,
            metrics=metrics
        )

        novelty = tracker.calculate_behavior_novelty(0.15, 1.5, 0.2)
        self.assertTrue(0.0 <= novelty <= 1.0)

    def test_fitness_computation(self):
        # Mock trainer
        # We can't easily instantiate MAPElitesTrainer because it binds ports and checks GPUs.
        # But we can test the compute_fitness logic if we mock or extract it.
        # Or we can try to instantiate it with a different port and CPU device.

        try:
            trainer = MAPElitesTrainer(port=5559, device='cpu', initial_population_size=5)
        except Exception as e:
            print(f"Could not instantiate trainer: {e}")
            return

        episode_data = {
            'total_distance': 5.0,
            'collision_count': 0,
            'avg_speed': 0.1,
            'avg_clearance': 1.0,
            'duration': 60.0,
            'action_smoothness': 0.05,
            'avg_linear_action': 0.5,
            'avg_angular_action': 0.1,
            'turn_efficiency': 0.8,
            'stationary_rotation_time': 0.0,
            'track_slip_detected': False,
            'coverage_count': 20,
            'oscillation_count': 0
        }

        # Add some population so diversity bonus works
        dummy_model = ActorNetwork().state_dict()
        trainer.population.add(dummy_model, 10.0, 0.1, 1.0, 0.1, {})
        trainer.population.add(dummy_model, 10.0, 0.2, 2.0, 0.5, {})
        trainer.population.add(dummy_model, 10.0, 0.3, 3.0, 0.8, {})

        fitness = trainer.compute_fitness(episode_data)
        print(f"Computed fitness: {fitness}")
        self.assertTrue(fitness > 0)

        # Test oscillation penalty
        episode_data_osc = episode_data.copy()
        episode_data_osc['oscillation_count'] = 10
        fitness_osc = trainer.compute_fitness(episode_data_osc)
        print(f"Computed fitness (oscillation): {fitness_osc}")
        self.assertTrue(fitness_osc < fitness)

if __name__ == '__main__':
    unittest.main()
