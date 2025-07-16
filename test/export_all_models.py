from .export_tc_model import export_tc_model

from ..model.cECG_CNN import cECG_CNN
from ..model.CET import CET_S
from ..model.ResTCN import ResTCN
from ..model.TEMPONet import TEMPONet
from ..model.tinychirp_cnn_time import TinyChirpCNNTime
from ..model.tinychirp_transformer_time import TinyChirpTransformerTime

from pathlib import Path



CLS_OF_MODELS = [
    cECG_CNN,
    CET_S,
    ResTCN,
    TEMPONet,
    TinyChirpTransformerTime,
    TinyChirpCNNTime
]

EXPORT_DIR = "./DeepSliding/artifact/"

if __name__ == "__main__":
    for i in range(0, 100, 10):
        overlap_r = i / 100
        export_path = EXPORT_DIR + f"r_{overlap_r}"
        Path(export_path).mkdir(parents=True, exist_ok=True)
        print(f"r_{overlap_r}")
        for cls in CLS_OF_MODELS:
            export_tc_model(export_path, cls, cls.DEFAULT_INPUT_SHAPE, int(cls.DEFAULT_INPUT_SHAPE[-1] * overlap_r))