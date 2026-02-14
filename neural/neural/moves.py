"""
Move Encoding/Decoding
======================

Bidirectional mapping between chess moves and policy vector indices.

The AlphaZero neural network outputs a policy vector of size 4672, where each
entry represents the probability of making a particular move. This module
converts between human-readable move representations (from-square, to-square,
optional promotion) and indices into this policy vector.

Policy Vector Layout: 8x8x73 = 4672
-------------------------------------

The policy vector is conceptually a 3D array indexed by:
    (source_rank, source_file, move_type)

where source_rank and source_file identify the origin square (0-7 each), and
move_type (0-72) identifies what kind of move is made from that square.

The flat index is computed as:
    index = (source_rank * 8 + source_file) * 73 + move_type

Move Types (73 total per source square)
----------------------------------------

**Queen moves (indices 0-55): 56 types**

"Queen moves" encode any move that a queen could make -- i.e., sliding along
one of 8 directions for 1 to 7 squares. This also covers rook moves, bishop
moves, pawn pushes, pawn captures, and queen promotions (which are encoded as
the underlying queen-type move, NOT as underpromotions).

    Direction indices:
        0 = North      (rank +, file  )   i.e., (+1, 0)
        1 = Northeast   (rank +, file +)   i.e., (+1, +1)
        2 = East        (rank  , file +)   i.e., (0, +1)
        3 = Southeast   (rank -, file +)   i.e., (-1, +1)
        4 = South       (rank -, file  )   i.e., (-1, 0)
        5 = Southwest   (rank -, file -)   i.e., (-1, -1)
        6 = West        (rank  , file -)   i.e., (0, -1)
        7 = Northwest   (rank +, file -)   i.e., (+1, -1)

    Distance indices: 1 to 7 (mapped to 0-6 internally)

    move_type = direction * 7 + (distance - 1)

    So the 56 queen-move indices are:
        0-6:   North, distances 1-7
        7-13:  Northeast, distances 1-7
        14-20: East, distances 1-7
        21-27: Southeast, distances 1-7
        28-34: South, distances 1-7
        35-41: Southwest, distances 1-7
        42-48: West, distances 1-7
        49-55: Northwest, distances 1-7

**Knight moves (indices 56-63): 8 types**

Knight moves are enumerated in a fixed order:

    Index 56: (+2, +1)   -- two ranks up, one file right
    Index 57: (+2, -1)   -- two ranks up, one file left
    Index 58: (+1, +2)   -- one rank up, two files right
    Index 59: (+1, -2)   -- one rank up, two files left
    Index 60: (-1, +2)   -- one rank down, two files right
    Index 61: (-1, -2)   -- one rank down, two files left
    Index 62: (-2, +1)   -- two ranks down, one file right
    Index 63: (-2, -1)   -- two ranks down, one file left

**Underpromotions (indices 64-72): 9 types**

When a pawn reaches the last rank, it can promote to a queen (handled as a
normal queen move above), or to a knight, bishop, or rook (underpromotions).
Each underpromotion is encoded by the combination of:
    - Piece type: knight (0), bishop (1), rook (2)
    - Direction: left-capture (0), straight push (1), right-capture (2)

    move_type = 64 + piece_index * 3 + direction_index

    Index 64: promote to knight, capturing left  (file - 1)
    Index 65: promote to knight, pushing straight (file unchanged)
    Index 66: promote to knight, capturing right  (file + 1)
    Index 67: promote to bishop, capturing left
    Index 68: promote to bishop, pushing straight
    Index 69: promote to bishop, capturing right
    Index 70: promote to rook, capturing left
    Index 71: promote to rook, pushing straight
    Index 72: promote to rook, capturing right

Board Flipping for Black's Perspective
---------------------------------------

The neural network always sees the board from the current player's perspective.
When it's black's turn, the board is flipped vertically in the encoding (see
encoding.py). The move encoding must match: before encoding a move for black,
we flip the ranks (rank r -> 7 - r) for both the source and destination
squares. This means:

    - "North" for black (toward rank 0 in real coordinates) becomes "North"
      in the flipped view (toward rank 7), matching what the network expects.
    - Promotions for black (reaching rank 0) become promotions on rank 7 in
      the flipped view, matching the white promotion convention.

The flip_for_black parameter in move_to_index and index_to_move controls this.

Usage
-----

    >>> from neural.moves import move_to_index, index_to_move, get_legal_move_mask
    >>> # White pawn push: e2 to e4 (rank 1 file 4 to rank 3 file 4)
    >>> idx = move_to_index(1, 4, 3, 4)
    >>> idx
    ... # some index in 0..4671
    >>> move = index_to_move(idx)
    >>> move
    (1, 4, 3, 4, None)
    >>> # Knight move: Ng1-f3 (rank 0 file 6 to rank 2 file 5)
    >>> idx = move_to_index(0, 6, 2, 5)
    >>> # Underpromotion: pawn on a7 promotes to knight on a8
    >>> idx = move_to_index(6, 0, 7, 0, promotion="knight")
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

# =============================================================================
# Constants
# =============================================================================

NUM_QUEEN_MOVE_TYPES: int = 56
"""7 distances x 8 directions = 56 queen-type move planes per source square."""

NUM_KNIGHT_MOVE_TYPES: int = 8
"""8 possible knight moves from any square."""

NUM_UNDERPROMOTION_TYPES: int = 9
"""3 piece types (knight, bishop, rook) x 3 directions (left, straight, right)."""

NUM_MOVE_TYPES: int = NUM_QUEEN_MOVE_TYPES + NUM_KNIGHT_MOVE_TYPES + NUM_UNDERPROMOTION_TYPES
"""Total move types per source square: 56 + 8 + 9 = 73."""
assert NUM_MOVE_TYPES == 73

BOARD_SIZE: int = 8
"""Chess board is 8x8."""

POLICY_SIZE: int = BOARD_SIZE * BOARD_SIZE * NUM_MOVE_TYPES
"""Total policy vector size: 64 squares x 73 move types = 4672."""
assert POLICY_SIZE == 4672

# --- Direction vectors for queen moves ---
# Each direction is a (delta_rank, delta_file) pair.
# The ordering matches the convention described in the module docstring.
QUEEN_DIRECTIONS: List[Tuple[int, int]] = [
    (+1, 0),   # 0: North
    (+1, +1),  # 1: Northeast
    (0, +1),   # 2: East
    (-1, +1),  # 3: Southeast
    (-1, 0),   # 4: South
    (-1, -1),  # 5: Southwest
    (0, -1),   # 6: West
    (+1, -1),  # 7: Northwest
]

# --- Knight move deltas ---
# The 8 possible knight moves, in the canonical order used for encoding.
KNIGHT_DELTAS: List[Tuple[int, int]] = [
    (+2, +1),  # index 0 (move_type 56)
    (+2, -1),  # index 1 (move_type 57)
    (+1, +2),  # index 2 (move_type 58)
    (+1, -2),  # index 3 (move_type 59)
    (-1, +2),  # index 4 (move_type 60)
    (-1, -2),  # index 5 (move_type 61)
    (-2, +1),  # index 6 (move_type 62)
    (-2, -1),  # index 7 (move_type 63)
]

# --- Underpromotion piece types ---
# The order is knight, bishop, rook. Queen promotion is NOT an underpromotion
# -- it's encoded as a regular queen move.
UNDERPROMOTION_PIECES: List[str] = ["knight", "bishop", "rook"]

# --- Underpromotion direction deltas (file change only) ---
# These represent the file change when a pawn promotes:
#   left-capture: file - 1
#   straight push: file unchanged
#   right-capture: file + 1
UNDERPROMOTION_FILE_DELTAS: List[int] = [-1, 0, +1]

# --- Offset constants for move_type ranges ---
QUEEN_MOVE_OFFSET: int = 0
"""Queen moves occupy move_type indices 0-55."""

KNIGHT_MOVE_OFFSET: int = NUM_QUEEN_MOVE_TYPES  # 56
"""Knight moves occupy move_type indices 56-63."""

UNDERPROMOTION_OFFSET: int = NUM_QUEEN_MOVE_TYPES + NUM_KNIGHT_MOVE_TYPES  # 64
"""Underpromotions occupy move_type indices 64-72."""


# =============================================================================
# Precomputed Lookup Tables
# =============================================================================
# For performance in hot loops, we precompute the inverse mappings so that
# index_to_move is O(1) rather than requiring search.

# _INDEX_TO_MOVE_TABLE[i] = (from_rank, from_file, to_rank, to_file, promotion)
# where promotion is None or a string like "knight", "bishop", "rook", "queen"
_INDEX_TO_MOVE_TABLE: List[Tuple[int, int, int, int, Optional[str]]] = []


def _build_index_to_move_table() -> List[Tuple[int, int, int, int, Optional[str]]]:
    """Build the lookup table mapping each policy index to its move components.

    This is called once at module load time. Each of the 4672 indices gets an
    entry, even if the resulting move would be off the board (those indices
    simply won't appear in legal move masks, but they exist for completeness
    so the mapping is total).

    Returns:
        List of 4672 tuples: (from_rank, from_file, to_rank, to_file, promotion).
        For off-board moves, to_rank or to_file may be outside 0-7.
    """
    table: List[Tuple[int, int, int, int, Optional[str]]] = []

    for from_rank in range(BOARD_SIZE):
        for from_file in range(BOARD_SIZE):
            # --- Queen moves (move_type 0-55) ---
            for direction_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
                for distance in range(1, 8):  # distances 1-7
                    to_rank = from_rank + dr * distance
                    to_file = from_file + df * distance
                    # Check if this is a pawn reaching the promotion rank
                    # via a queen-type move -- that's a queen promotion.
                    promotion: Optional[str] = None
                    if to_rank == 7 and from_rank == 6:
                        # A piece moving from rank 6 to rank 7 could be a
                        # promoting pawn. In the canonical (flipped-for-black)
                        # view, promotion always happens on rank 7.
                        # Queen-type moves to rank 7 from rank 6 are queen
                        # promotions (if the move is legal for a pawn).
                        # We encode "queen" so the caller knows.
                        if distance == 1 and abs(df) <= 1:
                            promotion = "queen"
                    table.append((from_rank, from_file, to_rank, to_file, promotion))

            # --- Knight moves (move_type 56-63) ---
            for dr, df in KNIGHT_DELTAS:
                to_rank = from_rank + dr
                to_file = from_file + df
                table.append((from_rank, from_file, to_rank, to_file, None))

            # --- Underpromotions (move_type 64-72) ---
            for piece_name in UNDERPROMOTION_PIECES:
                for file_delta in UNDERPROMOTION_FILE_DELTAS:
                    # Underpromotions always go from rank 6 to rank 7
                    # (in the canonical view where the current player's
                    # promotion rank is 7).
                    to_rank = from_rank + 1  # one rank forward
                    to_file = from_file + file_delta
                    table.append((from_rank, from_file, to_rank, to_file, piece_name))

    return table


_INDEX_TO_MOVE_TABLE = _build_index_to_move_table()
assert len(_INDEX_TO_MOVE_TABLE) == POLICY_SIZE


# =============================================================================
# Encoding: Move -> Index
# =============================================================================


def move_to_index(
    from_rank: int,
    from_file: int,
    to_rank: int,
    to_file: int,
    promotion: Optional[str] = None,
    flip_for_black: bool = False,
) -> int:
    """Map a chess move to a policy vector index (0..4671).

    The move is specified by source and destination squares plus an optional
    promotion piece. The encoding follows the AlphaZero convention:

    1. If the move is an underpromotion (to knight, bishop, or rook), it's
       encoded using the underpromotion planes (indices 64-72 per square).
    2. If the move matches a knight move pattern, it's encoded as a knight
       move (indices 56-63 per square).
    3. Otherwise, it's encoded as a queen-type move (indices 0-55 per square).
       This covers all sliding moves (rook, bishop, queen) and pawn moves.
       Queen promotions are also encoded this way (as a regular queen-type
       move to the promotion rank).

    Args:
        from_rank: Source rank (0-7, where 0 = rank 1).
        from_file: Source file (0-7, where 0 = a-file).
        to_rank: Destination rank (0-7).
        to_file: Destination file (0-7).
        promotion: Promotion piece type as a string: "queen", "knight",
            "bishop", or "rook". None for non-promotion moves. Queen
            promotions may also pass None (they're encoded as queen moves).
        flip_for_black: If True, flip the ranks before encoding. Set this
            to True when encoding moves for black's turn, to match the
            board flipping done in the neural network input encoding.

    Returns:
        An integer index in [0, 4671].

    Raises:
        ValueError: If the move cannot be encoded (e.g., invalid coordinates,
            or the delta doesn't match any known move pattern).

    Examples:
        >>> # White pawn e2-e4
        >>> move_to_index(1, 4, 3, 4)
        ... # queen-type move: North, distance 2, from square (1, 4)
        >>> # Knight Ng1-f3
        >>> move_to_index(0, 6, 2, 5)
        ... # knight move from (0, 6)
        >>> # Black pawn promotes to knight: a2-a1 (black's perspective, pre-flip)
        >>> move_to_index(1, 0, 0, 0, promotion="knight", flip_for_black=True)
        ... # After flip: from (6,0) to (7,0), underpromotion to knight, straight
    """
    # --- Step 1: Apply board flip for black ---
    if flip_for_black:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    # --- Step 2: Compute the delta ---
    dr = to_rank - from_rank
    df = to_file - from_file

    # --- Step 3: Determine the move type ---

    # Case A: Underpromotion (knight, bishop, or rook promotion)
    if promotion is not None and promotion != "queen":
        if promotion not in UNDERPROMOTION_PIECES:
            raise ValueError(
                f"Invalid promotion piece: '{promotion}'. "
                f"Expected one of: {UNDERPROMOTION_PIECES + ['queen']} or None."
            )
        piece_index = UNDERPROMOTION_PIECES.index(promotion)

        if df not in UNDERPROMOTION_FILE_DELTAS:
            raise ValueError(
                f"Invalid underpromotion file delta: {df}. "
                f"Expected -1 (left capture), 0 (push), or +1 (right capture)."
            )
        direction_index = UNDERPROMOTION_FILE_DELTAS.index(df)

        move_type = UNDERPROMOTION_OFFSET + piece_index * 3 + direction_index

    # Case B: Knight move
    elif _is_knight_move(dr, df):
        knight_index = KNIGHT_DELTAS.index((dr, df))
        move_type = KNIGHT_MOVE_OFFSET + knight_index

    # Case C: Queen-type move (including queen promotion)
    else:
        direction_index = _delta_to_queen_direction(dr, df)
        distance = max(abs(dr), abs(df))

        if distance < 1 or distance > 7:
            raise ValueError(
                f"Invalid queen-move distance: {distance}. "
                f"Move from ({from_rank},{from_file}) to ({to_rank},{to_file}) "
                f"has delta ({dr},{df})."
            )

        move_type = QUEEN_MOVE_OFFSET + direction_index * 7 + (distance - 1)

    # --- Step 4: Compute flat index ---
    square_index = from_rank * BOARD_SIZE + from_file
    index = square_index * NUM_MOVE_TYPES + move_type

    return index


# =============================================================================
# Decoding: Index -> Move
# =============================================================================


def index_to_move(
    index: int,
    flip_for_black: bool = False,
) -> Tuple[int, int, int, int, Optional[str]]:
    """Map a policy vector index back to move components.

    This is the inverse of move_to_index. Given a flat index into the policy
    vector, it returns the source square, destination square, and optional
    promotion piece.

    Args:
        index: Policy vector index in [0, 4671].
        flip_for_black: If True, flip the ranks in the output to convert
            from the network's canonical view back to the real board
            coordinates for black's turn.

    Returns:
        A tuple (from_rank, from_file, to_rank, to_file, promotion) where:
            - from_rank, from_file: Source square (0-7 each).
            - to_rank, to_file: Destination square (may be off-board for
              some indices that correspond to impossible moves).
            - promotion: None for non-promotions, or a string ("queen",
              "knight", "bishop", "rook") for promotion moves.

    Raises:
        ValueError: If index is out of range [0, 4671].
    """
    if not (0 <= index < POLICY_SIZE):
        raise ValueError(
            f"Policy index must be in [0, {POLICY_SIZE - 1}], got {index}."
        )

    from_rank, from_file, to_rank, to_file, promotion = _INDEX_TO_MOVE_TABLE[index]

    if flip_for_black:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    return (from_rank, from_file, to_rank, to_file, promotion)


# =============================================================================
# Legal Move Masking
# =============================================================================


def get_legal_move_mask(
    legal_moves: Sequence[Tuple[int, int, int, int, Optional[str]]],
    flip_for_black: bool = False,
) -> torch.Tensor:
    """Create a boolean mask over the policy vector for legal moves.

    Given a list of legal moves, returns a tensor of shape (4672,) where
    True indicates a legal move and False indicates an illegal move. This
    mask is used to zero out illegal move probabilities in the policy output
    before normalization.

    Args:
        legal_moves: Sequence of (from_rank, from_file, to_rank, to_file,
            promotion) tuples representing all legal moves in the position.
            Each move uses real board coordinates (not flipped).
        flip_for_black: If True, flip ranks before encoding (set this when
            it's black's turn).

    Returns:
        A boolean tensor of shape (4672,) with True for legal moves.
    """
    mask = torch.zeros(POLICY_SIZE, dtype=torch.bool)

    for move in legal_moves:
        from_rank, from_file, to_rank, to_file, promotion = move
        idx = move_to_index(
            from_rank, from_file, to_rank, to_file,
            promotion=promotion,
            flip_for_black=flip_for_black,
        )
        mask[idx] = True

    return mask


# =============================================================================
# Helper Functions
# =============================================================================


def _is_knight_move(dr: int, df: int) -> bool:
    """Check if a (delta_rank, delta_file) pair is a knight move.

    A knight move has one component with absolute value 1 and the other
    with absolute value 2.

    Args:
        dr: Rank delta.
        df: File delta.

    Returns:
        True if (dr, df) is a valid knight move delta.
    """
    return (abs(dr), abs(df)) in {(2, 1), (1, 2)}


def _delta_to_queen_direction(dr: int, df: int) -> int:
    """Convert a move delta to a queen direction index.

    The delta must be along one of the 8 cardinal/diagonal directions.
    The direction is determined by the signs of dr and df (the magnitude
    doesn't matter -- that's the distance).

    Args:
        dr: Rank delta (non-zero for N/S/diagonal moves).
        df: File delta (non-zero for E/W/diagonal moves).

    Returns:
        Direction index (0-7) matching the QUEEN_DIRECTIONS table.

    Raises:
        ValueError: If the delta is not along a queen direction (e.g.,
            a knight move or zero delta).
    """
    if dr == 0 and df == 0:
        raise ValueError("Zero delta is not a valid queen move.")

    # Validate that the delta is along one of the 8 queen directions.
    # For cardinal directions (N/S/E/W), one component must be 0.
    # For diagonal directions (NE/SE/SW/NW), both components must have
    # the same absolute value.
    abs_dr = abs(dr)
    abs_df = abs(df)
    if abs_dr != 0 and abs_df != 0 and abs_dr != abs_df:
        raise ValueError(
            f"Delta ({dr}, {df}) is not along a queen direction. "
            f"|dr|={abs_dr} != |df|={abs_df}, so this is neither a cardinal "
            f"nor a diagonal move. This might be a knight move."
        )

    # Normalize to unit direction
    # For diagonal moves, |dr| == |df|. For straight moves, one is 0.
    sign_r = (1 if dr > 0 else (-1 if dr < 0 else 0))
    sign_f = (1 if df > 0 else (-1 if df < 0 else 0))

    unit_direction = (sign_r, sign_f)

    # Look up the direction index
    for idx, direction in enumerate(QUEEN_DIRECTIONS):
        if direction == unit_direction:
            return idx

    raise ValueError(
        f"Delta ({dr}, {df}) is not along a queen direction. "
        f"Unit direction ({sign_r}, {sign_f}) not found in QUEEN_DIRECTIONS. "
        f"This might be a knight move -- check with _is_knight_move first."
    )
