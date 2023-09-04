import torch
from einops import repeat


def get_node_feat_convertor(mode: str):
    """
    Get the path feature convertor based on the mode
    args:
        mode: the mode of path feature convertor
    return:
        path_feat_convertor: the path feature convertor
    """
    if mode in ["MoMd", "momd"]:
        return convert_nf_to_MoMd_feat
    elif mode in ["SoMd", "somd"]:
        return convert_nf_to_SoMd_feat
    elif mode in ["SoSd", "sosd"]:
        return convert_nf_to_SoSd_feat
    else:
        raise ValueError(f"Invalid mode: {mode}")


def convert_nf_to_MoMd_feat(
    batch_node_feat: torch.Tensor,
    batch_size: int,
):
    """
    Convert node features from nf to Multi-origin Multi-destination feature format

    args:
        batch_node_feat: node features (batch x num. nodes, feat. dim)
        batch_size: batch size

    return:
        MoMd_feat: Multi-origin Multi-destination feature format (batch, #. nodes, #. nodes, feat. dim)

    """

    b = batch_size
    n = batch_node_feat.shape[0] // b
    assert (
        batch_node_feat.shape[0] % b == 0
    ), "Number of total nodes must be divisible by batch size"
    d = batch_node_feat.shape[-1]

    nf = batch_node_feat.view(b, n, d)  # [batch, num. nodes, dim]
    n_i = repeat(nf, "b n d -> b n m d", m=n)
    n_j = repeat(nf, "b n d -> b m n d", m=n)
    MoMd_feat = torch.cat([n_i, n_j], dim=-1)  # [batch, num. nodes, num. nodes, 2*dim]
    return MoMd_feat


def convert_nf_to_SoMd_feat(batch_node_feat: torch.Tensor, batch_size: int):
    """
    Convert node features from nf to Single-origin Multi-destination feature format

    args:
        batch_node_feat: node features (batch x num. nodes, feat. dim)
        batch_size: batch size

    return:
        SoMd_feat: Single-origin Multi-destination feature format (batch, num. nodes, feat. dim)

    """

    b = batch_size
    n = batch_node_feat.shape[0] // b
    assert (
        batch_node_feat.shape[0] % b == 0
    ), "Number of total nodes must be divisible by batch size"
    d = batch_node_feat.shape[-1]

    nf = batch_node_feat.view(b, n, d)  # [batch, num. nodes, dim]
    n_0 = repeat(nf[:, 0, :], "b d -> b n d", n=n)  # [batch, num. nodes, dim]
    SoMd_feat = torch.cat([n_0, nf], dim=-1)  # [batch, num. nodes, 2*dim]
    return SoMd_feat


def convert_nf_to_SoSd_feat(batch_node_feat: torch.Tensor, batch_size: int):
    """
    Convert node features from nf to Single-origin Single-destination feature format

    args:
        batch_node_feat: node features (batch x num. nodes, feat. dim)
        batch_size: batch size
    return:
        SoSd_feat: Single-origin Single-destination feature format (batch, 2*feat. dim)
    """
    b = batch_size
    n = batch_node_feat.shape[0] // b
    assert (
        batch_node_feat.shape[0] % b == 0
    ), "Number of total nodes must be divisible by batch size"
    d = batch_node_feat.shape[-1]

    nf = batch_node_feat.view(b, n, d)  # [batch, num. nodes, feat dim]
    SoSd_feat = torch.cat([nf[:, 0, :], nf[:, -1, :]], dim=-1)  # [batch, 2* feat dim]
    return SoSd_feat


def get_path_feat_convertor(mode: str):
    """
    Get the path feature convertor based on the mode
    args:
        mode: the mode of path feature convertor
    return:
        path_feat_convertor: the path feature convertor
    """
    if mode in ["MoMd", "momd"]:
        return convert_pf_to_MoMd_feat
    elif mode in ["SoMd", "somd"]:
        return convert_pf_to_SoMd_feat
    elif mode in ["SoSd", "sosd"]:
        return convert_pf_to_SoSd_feat
    else:
        raise ValueError(f"Invalid mode: {mode}")


def convert_pf_to_MoMd_feat(path_feat: torch.Tensor):
    """
    Convert path features from pf to Multi-origin Multi-destination feature format
    !!! Serve as a placeholder !!!
    args:
        path_feat: node features (batch, #. nodes, # nodes, feat. dim)
        batch_size: batch size

    return:
        MoMd_feat: Multi-origin Multi-destination feature format (batch, #. nodes, #. nodes, feat. dim)
    """

    # !!! Serve as a placeholder for consistency with the other modules !!!
    return path_feat


def convert_pf_to_SoMd_feat(path_feat: torch.Tensor):
    return path_feat[
        :, 0, :, :
    ]  # assume the first node is the origin -- (batch, num. nodes, feat. dim)


def convert_pf_to_SoSd_feat(path_feat: torch.Tensor):
    return path_feat[
        :, 0, -1, :
    ]  # assume the first and last node are the origin and destination -- (batch, feat. dim)
