from .aespt import load_aespt
from .cdpa import load_cdpa
from .sakura_aes import load_sakura_aes
from .ches_challenge import load_ches_challenge

from .preprocess import *

DATA_TYPE = {
    'aespt': load_aespt,
    'xmega': load_cdpa, 
    'xmega_em': load_cdpa,
    'sakura_aes': load_sakura_aes,
    'ches_challenge': load_ches_challenge
}

PREPROCESS_METHOD = {
    "no_preprocess": no_preprocessing, 
    "standardize": feature_standardization,
    'shift': shift
}