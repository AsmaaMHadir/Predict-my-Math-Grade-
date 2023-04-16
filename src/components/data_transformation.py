import os
import sys
sys.path.insert(0,'/Users/owner/Downloads/mlproject/src')

from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass