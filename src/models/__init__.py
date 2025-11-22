"""
Models module
Contains machine learning models for track performance prediction
"""

from .transfer_learning_model import (
    TrackTransferModel,
    SourceTrackEncoder,
    TrackDNAEncoder,
    TransferNetwork,
    PerformancePredictor,
    TransferLearningTrainer,
    create_transfer_model
)
from .transfer_data_preparer import TransferDataPreparer
from .track_coach import TrackCoach, create_track_coach
from .track_coaches.barber_coach import BarberCoach, create_barber_coach
from .track_coaches.cota_coach import COTACoach, create_cota_coach
from .track_coaches.indy_coach import IndianapolisCoach, create_indianapolis_coach
from .track_coaches.vir_coach import VIRCoach, create_vir_coach

__all__ = [
    'TrackTransferModel',
    'SourceTrackEncoder',
    'TrackDNAEncoder',
    'TransferNetwork',
    'PerformancePredictor',
    'TransferLearningTrainer',
    'create_transfer_model',
    'TransferDataPreparer',
    'TrackCoach',
    'create_track_coach',
    'BarberCoach',
    'create_barber_coach',
    'COTACoach',
    'create_cota_coach',
    'IndianapolisCoach',
    'create_indianapolis_coach',
    'VIRCoach',
    'create_vir_coach'
]
