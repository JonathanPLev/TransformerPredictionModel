#!/usr/bin/env python3
"""
Generate training data for transformer-based NBA player performance prediction.
Creates sequences with lag features, seasonal averages, and contextual data.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlalchemy as sqla
from src.server.db_local import engine
from typing import Dict, List, Tuple, Optional
import pickle

class DataPreparer():
    def __init__(self):
        