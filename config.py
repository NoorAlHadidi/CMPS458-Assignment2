HYPERPARAMS = {
    'gamma': 0.99,
    'epsilon_start': 1,
    'epsilon_end': 0.01,
    'epsilon_decay': 2500,
    'learning_rate': 0.001,
    'memory_size': 10000,
    'batch_size': 64,
    'tau': 0.005
}

CONFIG = {
    'CartPole-v1': {
        'max_episodes': 500,
        'success_threshold': 475,
        'hyperparams': {
            **HYPERPARAMS,
            'epsilon_decay': 2000,
            'learning_rate': 0.0007
        }
    },
    'Acrobot-v1': {
        'max_episodes': 1000,
        'success_threshold': -100,
        'hyperparams': {
            **HYPERPARAMS
        }
    },
    'MountainCar-v0': {
        'max_episodes': 1000,
        'success_threshold': -110,
        'hyperparams': {
            **HYPERPARAMS
        }
    },
    'Pendulum-v1': {
        'max_episodes': 500,
        'success_threshold': -200,
        'hyperparams': {
            **HYPERPARAMS
        }
    }
}
