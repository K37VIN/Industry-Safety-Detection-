import os, sys
from six.moves import urllib
from isd.logger import logging
from isd.exception import isdException
from isd.entity.config_entity import ModelTrainerConfig
from isd.entity.artifact_entity import ModelTrainerArtifact
import shutil 

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
    ):
        self.model_trainer_config = model_trainer_config

    