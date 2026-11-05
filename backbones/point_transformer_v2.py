from __future__ import annotations

from networks.point_transv2_carla import PointTransformerV2


def build_backbone(opt):
    grid_sizes = build_grid_sizes(opt)
    return PointTransformerV2(
        in_channels=opt["in_dim"],
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=opt["enc_depth"],
        enc_channels=opt["enc_channels"],
        enc_groups=opt["enc_groups"],
        enc_neighbours=[opt["num_neighbors"] for _ in range(opt["num_layers"])],
        dec_depths=opt["dec_depth"],
        dec_channels=opt["dec_channels"],
        dec_groups=opt["dec_groups"],
        dec_neighbours=[opt["num_neighbors"] for _ in range(opt["num_layers"])],
        grid_sizes=grid_sizes,
        attn_qkv_bias=True,
        pe_multiplier=opt["pe_mul"],
        pe_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0,
        enable_checkpoint=False,
        unpool_backend=opt["unpool"],
        get_2nd=opt.get("get_2nd", False),
    )


def build_grid_sizes(opt):
    grid_sizes = []
    grid = opt["init_grid"]
    num_layers = opt["num_layers"]
    if opt["grid_first"]:
        grid_sizes.append(grid)
        num_layers = num_layers - 1
    for i in range(num_layers):
        grid = grid * opt["sub_sampling_ratio"][i]
        grid_sizes.append(grid)
    return grid_sizes
