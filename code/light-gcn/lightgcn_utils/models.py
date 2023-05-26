import os

import torch
from torch_geometric.nn.models import LightGCN
from .utils import get_logger, logging_conf


logger = get_logger(logger_conf=logging_conf)


########################   Model Build
def build(n_node: int, weight: str = None, **kwargs):
    """
    Args:
        n_node (int): The number of nodes in the graph.
        **kwargs (optional)
        - embedding_dim (int): The dimensionality of node embeddings.
        - num_layers (int): The number of
            :class:`~torch_geometric.nn.conv.LGConv` layers.
        - alpha (float or torch.Tensor, optional): The scalar or vector
            specifying the re-weighting coefficients for aggregating the final
            embedding. If set to :obj:`None`, the uniform initialization of
            :obj:`1 / (num_layers + 1)` is used. (default: :obj:`None`)
    Returns:
        model
    """

    model = LightGCN(num_nodes=n_node, **kwargs)
    if weight:  # 사전에 학습된 모델 가중치의 파일 경로가 있으면
        if not os.path.isfile(path=weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(f=weight)["model"]  # weight 경로에 저장된 모델 가중치 파일 로드
        model.load_state_dict(state)  # 저장된 모델 가중치를 model에 로드. 모델의 가중치 업데이트
        return model  # 가중치가 적용된 모델
    else:  # 가중치 파일이 지정되지 않은 경우에는 모델을 그대로 반환하고, 로그 메시지를 출력
        logger.info("No load model")
        return model
