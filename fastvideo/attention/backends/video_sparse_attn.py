# SPDX-License-Identifier: Apache-2.0
import functools
import math
from dataclasses import dataclass

import torch

import fastvideo.envs as envs

try:
    from fastvideo_kernel import video_sparse_attn
except ImportError:
    video_sparse_attn = None
try:
    from fastvideo_kernel import video_sparse_attn_bshd
except ImportError:
    video_sparse_attn_bshd = None

from typing import Any

from fastvideo.attention.backends.abstract import (AttentionBackend, AttentionImpl, AttentionMetadata,
                                                   AttentionMetadataBuilder)
from fastvideo.distributed import get_sp_group
from fastvideo.logger import init_logger

logger = init_logger(__name__)
# VSA tile shape. The tile volume picks the kernel path in forward():
# (4,4,4)=64 -> existing TK/Triton path (default, unchanged);
# (4,8,8)=256 -> FA4 CuTe block-sparse attention fastpath (Blackwell).
DEFAULT_VSA_TILE_SIZE = (4, 4, 4)
BSHD_VSA_TILE_SIZE = (4, 8, 8)
VSA_TILE_SIZE = DEFAULT_VSA_TILE_SIZE
_LOGGED_TILE_SIZES: set[tuple[int, int, int]] = set()


def _parse_tile_size(value: str) -> tuple[int, int, int]:
    normalized = value.strip().lower().replace("x", ",")
    parts = tuple(int(part.strip()) for part in normalized.split(",") if part.strip())
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"FASTVIDEO_VSA_TILE_SIZE must be 'auto' or three positive ints, got {value!r}")
    return parts


def _select_vsa_tile_size(requested_tile_size: list[int] | tuple[int, int, int] | str | None) -> tuple[int, int, int]:
    if requested_tile_size is None:
        requested_tile_size = envs.FASTVIDEO_VSA_TILE_SIZE
    if requested_tile_size is None or requested_tile_size == "":
        return DEFAULT_VSA_TILE_SIZE
    if isinstance(requested_tile_size, list | tuple):
        tile_size = tuple(int(part) for part in requested_tile_size)
        if len(tile_size) != 3 or any(part <= 0 for part in tile_size):
            raise ValueError(f"VSA_tile_size must contain three positive ints, got {requested_tile_size!r}")
    else:
        normalized = requested_tile_size.strip().lower()
        if normalized in {"default", "legacy"}:
            tile_size = DEFAULT_VSA_TILE_SIZE
        elif normalized == "auto":
            tile_size = BSHD_VSA_TILE_SIZE if video_sparse_attn_bshd is not None else DEFAULT_VSA_TILE_SIZE
        else:
            tile_size = _parse_tile_size(requested_tile_size)

    if math.prod(tile_size) == 256 and video_sparse_attn_bshd is None:
        logger.warning(
            "FASTVIDEO_VSA_TILE_SIZE=%s requested the BSHD fast path, but video_sparse_attn_bshd is not "
            "installed; falling back to %s.", tile_size, DEFAULT_VSA_TILE_SIZE)
        return DEFAULT_VSA_TILE_SIZE
    return tile_size


def _log_vsa_tile_size(tile_size: tuple[int, int, int]) -> None:
    if tile_size in _LOGGED_TILE_SIZES:
        return
    _LOGGED_TILE_SIZES.add(tile_size)
    logger.info("Using VSA tile size %s (%d tokens per block)", tile_size, math.prod(tile_size))


@functools.lru_cache(maxsize=10)
def get_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    T, H, W = dit_seq_shape
    ts, hs, ws = tile_size
    indices = torch.arange(T * H * W, device=device, dtype=torch.long).reshape(T, H, W)
    ls = []
    for t in range(math.ceil(T / ts)):
        for h in range(math.ceil(H / hs)):
            for w in range(math.ceil(W / ws)):
                ls.append(indices[t * ts:min(t * ts + ts, T), h * hs:min(h * hs + hs, H),
                                  w * ws:min(w * ws + ws, W)].flatten())
    index = torch.cat(ls, dim=0)
    return index


@functools.lru_cache(maxsize=10)
def get_reverse_tile_partition_indices(
    dit_seq_shape: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    return torch.argsort(get_tile_partition_indices(dit_seq_shape, tile_size, device))


@functools.lru_cache(maxsize=10)
def construct_variable_block_sizes(
    dit_seq_shape: tuple[int, int, int],
    num_tiles: tuple[int, int, int],
    tile_size: tuple[int, int, int],
    device: torch.device,
) -> torch.LongTensor:
    """
    Compute the number of valid (non‑padded) tokens inside every
    (ts_t × ts_h × ts_w) tile after padding ‑‑ flattened in the order
    (t‑tile, h‑tile, w‑tile) that `rearrange` uses.

    Returns
    -------
    torch.LongTensor  # shape: [∏ full_window_size]
    """
    # unpack
    t, h, w = dit_seq_shape
    ts_t, ts_h, ts_w = tile_size
    n_t, n_h, n_w = num_tiles

    def _sizes(dim_len: int, tile: int, n_tiles: int) -> torch.LongTensor:
        """Vector with the size of each tile along one dimension."""
        sizes = torch.full((n_tiles, ), tile, dtype=torch.int, device=device)
        # size of last (possibly partial) tile
        remainder = dim_len - (n_tiles - 1) * tile
        sizes[-1] = remainder if remainder > 0 else tile
        return sizes

    t_sizes = _sizes(t, ts_t, n_t)  # [n_t]
    h_sizes = _sizes(h, ts_h, n_h)  # [n_h]
    w_sizes = _sizes(w, ts_w, n_w)  # [n_w]

    # broadcast‑multiply to get voxels per tile, then flatten
    block_sizes = (
        t_sizes[:, None, None]  # [n_t, 1,   1]
        * h_sizes[None, :, None]  # [1,   n_h, 1]
        * w_sizes[None, None, :]  # [1,   1,   n_w]
    ).reshape(-1)  # [n_t * n_h * n_w]

    return block_sizes


@functools.lru_cache(maxsize=10)
def get_non_pad_index(
    variable_block_sizes: torch.LongTensor,
    max_block_size: int,
):
    n_win = variable_block_sizes.shape[0]
    device = variable_block_sizes.device
    starts_pad = torch.arange(n_win, device=device) * max_block_size
    index_pad = starts_pad[:, None] + torch.arange(max_block_size, device=device)[None, :]
    index_mask = torch.arange(max_block_size, device=device)[None, :] < variable_block_sizes[:, None]
    return index_pad[index_mask]


class VideoSparseAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @staticmethod
    def get_name() -> str:
        return "VIDEO_SPARSE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["VideoSparseAttentionImpl"]:
        return VideoSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["VideoSparseAttentionMetadata"]:
        return VideoSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["VideoSparseAttentionMetadataBuilder"]:
        return VideoSparseAttentionMetadataBuilder


@dataclass
class VideoSparseAttentionMetadata(AttentionMetadata):
    current_timestep: int
    dit_seq_shape: list[int]
    tile_size: tuple[int, int, int]
    num_tiles: list[int]
    total_seq_length: int
    tile_partition_indices: torch.LongTensor
    reverse_tile_partition_indices: torch.LongTensor
    variable_block_sizes: torch.LongTensor
    non_pad_index: torch.LongTensor
    # Precomputed fancy index that fuses ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
    # in postprocess_output().  Avoids materializing the intermediate
    # ``[B, len(non_pad_index), H, D]`` tensor on every layer.
    untile_combined_index: torch.LongTensor
    # Per-step shared padded buffer used by tile().  Lazily populated on
    # the first layer's call and reused by every subsequent VSA layer in
    # the same denoising step.  Scoping to metadata (not class/instance)
    # makes the reuse thread-safe across concurrent requests and keeps
    # the "pad positions are zero" invariant trivially true (the buffer
    # is freshly zeroed alongside ``non_pad_index`` so the index set
    # cannot drift between calls).
    tile_buf: torch.Tensor | None = None


class VideoSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self) -> None:
        pass

    def prepare(self) -> None:
        pass

    def build(  # type: ignore
        self,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        VSA_sparsity: float,
        device: torch.device,
        VSA_tile_size: list[int] | tuple[int, int, int] | str | None = None,
        **kwargs: dict[str, Any],
    ) -> VideoSparseAttentionMetadata:
        patch_size = patch_size
        dit_seq_shape = (raw_latent_shape[0] // patch_size[0], raw_latent_shape[1] // patch_size[1],
                         raw_latent_shape[2] // patch_size[2])
        tile_size = _select_vsa_tile_size(VSA_tile_size)
        _log_vsa_tile_size(tile_size)

        num_tiles = (math.ceil(dit_seq_shape[0] / tile_size[0]), math.ceil(dit_seq_shape[1] / tile_size[1]),
                     math.ceil(dit_seq_shape[2] / tile_size[2]))
        total_seq_length = math.prod(dit_seq_shape)

        tile_partition_indices = get_tile_partition_indices(dit_seq_shape, tile_size, device)
        reverse_tile_partition_indices = get_reverse_tile_partition_indices(dit_seq_shape, tile_size, device)
        variable_block_sizes = construct_variable_block_sizes(dit_seq_shape, num_tiles, tile_size, device)
        non_pad_index = get_non_pad_index(variable_block_sizes, math.prod(tile_size))
        untile_combined_index = non_pad_index[reverse_tile_partition_indices]

        return VideoSparseAttentionMetadata(
            current_timestep=current_timestep,
            dit_seq_shape=dit_seq_shape,  # type: ignore
            tile_size=tile_size,
            VSA_sparsity=VSA_sparsity,  # type: ignore
            num_tiles=num_tiles,  # type: ignore
            total_seq_length=total_seq_length,  # type: ignore
            tile_partition_indices=tile_partition_indices,  # type: ignore
            reverse_tile_partition_indices=reverse_tile_partition_indices,
            variable_block_sizes=variable_block_sizes,
            non_pad_index=non_pad_index,
            untile_combined_index=untile_combined_index)


class VideoSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.prefix = prefix
        sp_group = get_sp_group()
        self.sp_size = sp_group.world_size

    def tile(self, x: torch.Tensor, attn_metadata: VideoSparseAttentionMetadata) -> torch.Tensor:
        """Tile ``x`` into ``attn_metadata.tile_buf`` and return it.

        The returned tensor aliases the per-metadata buffer and is only
        valid until the next ``tile()`` / ``preprocess_qkv`` call on the
        same ``attn_metadata``.  Callers must consume (or copy) the
        result before invoking another VSA layer with the same metadata.
        Today both call sites materialize copies via
        ``.transpose(...).contiguous()`` inside ``forward()``, so the
        contract holds; future callers must preserve it.
        """
        num_tiles = attn_metadata.num_tiles
        tile_size = attn_metadata.tile_size
        t_padded_size = num_tiles[0] * tile_size[0]
        h_padded_size = num_tiles[1] * tile_size[1]
        w_padded_size = num_tiles[2] * tile_size[2]
        target_shape = (x.shape[0], t_padded_size * h_padded_size * w_padded_size, x.shape[-2], x.shape[-1])

        # Reuse the per-step buffer stashed on metadata (lazily allocated
        # on the first VSA layer's call within a denoising step).  Pad
        # positions are zero from the initial torch.zeros and never
        # written to.  Scoping to metadata makes reuse safe across
        # concurrent requests and keeps the "pad positions are zero"
        # invariant trivially true: ``non_pad_index`` is fixed within
        # a single metadata instance.
        buf = attn_metadata.tile_buf
        if (buf is None or buf.shape != target_shape or buf.dtype != x.dtype or buf.device != x.device):
            buf = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
            attn_metadata.tile_buf = buf

        buf[:, attn_metadata.non_pad_index] = x[:, attn_metadata.tile_partition_indices]
        return buf

    def untile(self, x: torch.Tensor, untile_combined_index: torch.LongTensor) -> torch.Tensor:
        # Single fancy index using precomputed combined indices; avoids
        # the intermediate ``[B, len(non_pad_index), H, D]`` tensor that
        # the two-step ``x[:, non_pad_index][:, reverse_tile_partition_indices]``
        # would allocate on every layer.
        return x[:, untile_combined_index]

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        """Tile QKV; aliasing contract: see ``tile()``."""
        return self.tile(qkv, attn_metadata)

    def postprocess_output(
        self,
        output: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        return self.untile(output, attn_metadata.untile_combined_index)

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate_compress: torch.Tensor,
        attn_metadata: VideoSparseAttentionMetadata,
    ) -> torch.Tensor:
        VSA_sparsity = attn_metadata.VSA_sparsity
        tile_size = attn_metadata.tile_size
        block_elements = math.prod(tile_size)
        cur_topk = math.ceil((1 - VSA_sparsity) * (attn_metadata.total_seq_length / block_elements))

        # 256-element tiles auto-route to the FA4 CuTe BSHD fastpath, which
        # consumes [B, S, H, D] directly -- skip the transpose round-trip.
        if block_elements == 256 and video_sparse_attn_bshd is not None:
            return video_sparse_attn_bshd(query,
                                          key,
                                          value,
                                          attn_metadata.variable_block_sizes,
                                          attn_metadata.variable_block_sizes,
                                          cur_topk,
                                          block_size=tile_size,
                                          compress_attn_weight=gate_compress)

        if video_sparse_attn is None:
            raise NotImplementedError("video_sparse_attn is not installed")
        # Default 64-element-tile path (unchanged): BHSD round-trip.
        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()
        gate_compress = gate_compress.transpose(1, 2).contiguous()
        return video_sparse_attn(query,
                                 key,
                                 value,
                                 attn_metadata.variable_block_sizes,
                                 attn_metadata.variable_block_sizes,
                                 cur_topk,
                                 block_size=tile_size,
                                 compress_attn_weight=gate_compress).transpose(1, 2)
