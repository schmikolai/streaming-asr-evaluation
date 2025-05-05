"""config file that reads all config from .env or CMD environment for app"""

import os

import yaml
from faster_whisper.tokenizer import _LANGUAGE_CODES
import argparse
import logging

logger = logging.getLogger(__name__)

def read_config(config_yml_path: str) -> dict:
    """Read the config from .env or environment variables, returns dict with config"""

    config = {}
    with open(config_yml_path, "r", encoding="utf-8") as data:
        config = yaml.safe_load(data)

    def get_config(key, default=None):
        """Function to check and get configuration"""

        # Get the value from the config file, if it is not there, get it from the environment variables
        # allow environment variables to override the config file for easy deployment
        value = config.get(key, os.getenv(key, default))
        if value is None:
            raise ValueError(
                f"Configuration error: '{key}' is not set in .env"
                + " as an environment variable or as a default value"
            )
        return value

    def get_extracted_field_from_config(key, nested_key):
        """Function to extract one field from an array of config options"""
        value = get_config(key)
        res = []
        for nested_val in value:
            if nested_key not in nested_val:
                raise ValueError(
                    f"Configuration error: '{nested_key}' could not be extracted from value from '{key}'"
                )
            res += nested_val[nested_key]
        return list(set(res))

    return {
        # Essential Configuration, these are required in config.yml
        "log_level": get_config("log_level").upper(),
        # Experiment Configuration
        "experiment": get_config("experiment"),
        # File System Configuration
        #   Path to the status file folder
        "status_file_path": get_config("status_file_path", default="data/status"),
        #   Path to the model folder
        "model_path": get_config("model_path", default="models"),
        #   Path to the audio file folder
        "audio_file_path": get_config("audio_file_path", default="data/audio_files"),
        #   Path to the audio file folder
        "export_file_path": get_config("audio_file_path", default="data/export"),
        #   Audio file format to use
        "audio_file_format": get_config("audio_file_format", default=".wav"),
        #
        # Cleanup Configuration
        #   Hours that status and audio files are kept
        "keep_data_for_hours": get_config("keep_data_for_hours", default=72),
        #   How often to clean up files in data (only runs if no transcriptions are in progress)
        "cleanup_schedule_in_minutes": get_config(
            "cleanup_schedule_in_minutes", default=10
        ),
        # Transcription default Configuration
        "transcription_default": get_config("transcription_default"),
        "supported_language_codes": list(_LANGUAGE_CODES),
    }

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", type=str, default="configs/default.yml", help="Path to the config file"
)

args = parser.parse_args()

# config_path = os.path.join(os.getcwd(), "configs/default.yml")
config_path = os.path.join(os.getcwd(), args.config)

logger.info(f"loading Config from {config_path}")

if os.path.exists(config_path):
    CONFIG = read_config(config_path)
    logger.info(f"Config loaded from {config_path}")
else:
    raise RuntimeWarning("No config file found")
