import os
import sys
import yaml

def load_config():
    """Load YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'detection_config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f'ERROR: Could not load configuration file: {e}')
        sys.exit(1)
