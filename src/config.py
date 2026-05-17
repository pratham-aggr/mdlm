import os

MODEL_ID   = "GSAI-ML/LLaDA-8B-Instruct"
GEN_LEN    = 200
STEPS      = 128
SEED       = 42
N_PROBLEMS = 200
N_ANALYSIS = 50

LAYERS_ALL      = list(range(32))        
LAYERS_MID_LATE = list(range(16, 32))    
LAYERS_EARLY    = list(range(0, 8))      
SAVE_LAYERS     = [0, 4, 8, 12, 16, 20, 24, 28, 31] 
SAVE_STEPS  = [0, 32, 64, 96, 127]

LOGICAL_CONNECTIVES = [
    "therefore", "thus", "so", "since",
    "hence", "then", "because", "giving"
]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "results")
DATA_DIR      = os.path.join(_PROJECT_ROOT, "data")
LAYER_PATH = "model.transformer.blocks"