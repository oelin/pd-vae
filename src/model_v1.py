from typing import List, Tuple
from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


# Transformer modules.

def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """RMS norm."""

    rms = x.square().mean(dim=-1, keepdim=True).sqrt()

    return x / rms


def linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:

    score = F.softmax(torch.einsum('bhnk,bhnc->bhkc', k/4, v), dim=-1)
    x = F.softmax(torch.einsum('bhnk,bhkc->bhnk', q/4, score), dim=-1)

    return x


class Attention(nn.Module):
    """Attention.

    Example
    -------
    >>> module = Attention(
    ...    embedding_dimension=256,
    ...    heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)
    """

    def __init__(
        self,
        *,
        embedding_dimension: int,
        heads: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.heads = heads

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q, k, v = rearrange(self.linear_1(x), 'b s (n h e) -> n b h s e', n=3, h=self.heads)
        x = linear_attention(q, k, v) #F.scaled_dot_product_attention(q, k, v)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))

        return x


class MLP(nn.Module):
    """MLP.

    Example
    -------
    >>> module = MLP(embedding_dimension=256)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        """

        super().__init__()

        self.linear_1 = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension * 3,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_dimension * 3,
            out_features=embedding_dimension,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = F.silu(self.linear_1(x))
        x = F.silu(self.linear_2(x))

        return x


class TransformerBlock(nn.Module):
    """Transformer block.

    Example
    -------
    >>> module = TransformerBlock(embedding_dimension=256, heads=16)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of heads.
        """

        super().__init__()

        self.attention = Attention(
            embedding_dimension=embedding_dimension,
            heads=heads,
        )

        self.mlp = MLP(embedding_dimension=embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = x + self.attention(rms_norm(x))
        x = x + self.mlp(rms_norm(x))

        return x


@dataclass(frozen=True)
class TransformerConfiguration:
    embedding_dimension: int
    heads: int
    blocks: int


class Transformer(nn.Module):
    """Transformer.

    Example
    -------
    >>> configuration = TransformerConfiguration(
    ...     embedding_dimenson=256,
    ...     heads=16,
    ...     blocks=16,
    ... )
    >>> module = Transformer(configuration=configuration)
    >>> x = torch.randn((1, 10, 256))
    >>> x = module(x)  # Shape: (1, 10, 256).
    """

    def __init__(self, *, configuration: TransformerConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : TransformerConfiguration
            The module configuration.
        """

        super().__init__()

        self.blocks = nn.Sequential(*[
            TransformerBlock(
                embedding_dimension=configuration.embedding_dimension,
                heads=configuration.heads,
            ) for _ in range(configuration.blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        x: torch.Tensor
            The output tensor.
        """

        x = self.blocks(x)

        return x


class VLVAE(nn.Module):

    def __init__(self, *, embedding_dimension: int, heads: int, layers: int, patch_size: int) -> None:
        super().__init__()

        self.encoder = Transformer(
            configuration=TransformerConfiguration(
                embedding_dimension=embedding_dimension,
                heads=heads,
                blocks=layers,
            ),
        )

        self.decoder = Transformer(
            configuration=TransformerConfiguration(
                embedding_dimension=embedding_dimension,
                heads=heads,
                blocks=layers,
            ),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embedding_dimension,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=1024,
            embedding_dim=embedding_dimension,
        )

        # self.patch_unembedding = nn.Sequential(
        #     nn.Upsample(scale_factor=patch_size),
        #     nn.Conv2d(
        #         in_channels=embedding_dimension,
        #         out_channels=3,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #     ),
        # )

        self.patch_unembedding = nn.ConvTranspose2d(
            in_channels=embedding_dimension,
            out_channels=3,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        p = self.position_embedding(torch.arange(x.size(-2), device=x.device))
        x = self.encoder(x + p)

        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:

        x = self.decoder(x)
        B, L, C = x.shape
        h = int(math.sqrt(L))

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=h)
        x = self.patch_unembedding(x)
        x = torch.sigmoid(x)

        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        z = self.encode(x)
        x = self.decode(z)

        return x, z
