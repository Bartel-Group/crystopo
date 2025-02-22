from dataclasses import dataclass

@dataclass
class MPConfig:
    """Configuration for Materials Project API interactions."""
    api_key: str = 'K3gsxpm6KIOixpgZzdvG5ofrPdbgG3lE'
    random_seed: int = 42  # For reproducibility
