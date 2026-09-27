from .assistant import Assistant
from .config import AssistantConfig


def main():
    config = AssistantConfig.load()
    Assistant(config).run()
