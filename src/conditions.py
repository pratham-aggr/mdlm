import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LAYERS_ALL, LAYERS_MID_LATE, LAYERS_EARLY

CONDITIONS = {
    'A': {
        'steer_layers':       [],
        'steer_positions':    'all',
        'steer_when':         'all',
        'save_hidden_states': True,
        'description':        'Baseline — no steering',
    },
    'B': {
        'steer_layers':       LAYERS_ALL,
        'steer_positions':    'all',
        'steer_when':         'all',
        'save_hidden_states': True,
        'description':        'Full steering — all layers, all positions, all steps',
    },
    'C': {
        'steer_layers':       LAYERS_ALL,
        'steer_positions':    'prompt_only',
        'steer_when':         'all',
        'save_hidden_states': False,
        'description':        'Prompt tokens only',
    },
    'D': {
        'steer_layers':       LAYERS_ALL,
        'steer_positions':    'response_only',
        'steer_when':         'all',
        'save_hidden_states': False,
        'description':        'Response tokens only',
    },
    'E': {
        'steer_layers':       LAYERS_ALL,
        'steer_positions':    'all',
        'steer_when':         'first_20pct',
        'save_hidden_states': False,
        'description':        'First 20% of denoising steps only',
    },
    'F': {
        'steer_layers':       LAYERS_ALL,
        'steer_positions':    'all',
        'steer_when':         'last_20pct',
        'save_hidden_states': False,
        'description':        'Last 20% of denoising steps only',
    },
    'G': {
        'steer_layers':       LAYERS_MID_LATE,
        'steer_positions':    'all',
        'steer_when':         'all',
        'save_hidden_states': False,
        'description':        'Mid-to-late layers only (16-32)',
    },
    'H': {
        'steer_layers':       LAYERS_EARLY,
        'steer_positions':    'all',
        'steer_when':         'all',
        'save_hidden_states': False,
        'description':        'Early layers only (0-8)',
    },
}