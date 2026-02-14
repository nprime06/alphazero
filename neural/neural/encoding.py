"""
Board Encoding
==============

Converts chess board states into neural network input tensors.

This module implements the input representation described in the AlphaZero paper
(Silver et al., 2018). Each board position is encoded as a stack of 8x8 binary
(or scalar) feature planes, where each plane represents one aspect of the
position (e.g., "where are white's pawns?").

Plane Layout (119 planes total)
-------------------------------

The 119 input planes are organized as follows:

**History planes (T=8 time steps, 14 planes each = 112 planes)**

For each of the T=8 most recent board positions (t=0 is the current position,
t=1 is the position one move ago, etc.):

    Offset within time step:
        0:  Current player's pawns      (1 where pawn exists, 0 elsewhere)
        1:  Current player's knights
        2:  Current player's bishops
        3:  Current player's rooks
        4:  Current player's queens
        5:  Current player's king
        6:  Opponent's pawns
        7:  Opponent's knights
        8:  Opponent's bishops
        9:  Opponent's rooks
        10: Opponent's queens
        11: Opponent's king
        12: Repetition count plane 1    (all 1s if position occurred >= 1 time before)
        13: Repetition count plane 2    (all 1s if position occurred >= 2 times before)

    Why "current player" and "opponent" instead of "white" and "black"?
        The network always sees the board from the perspective of the player to
        move. When it's black's turn, the board is flipped vertically so that
        black's pieces appear where white's would normally be. This means the
        network only needs to learn one set of patterns, halving the effective
        complexity.

    Why 2 repetition planes?
        Threefold repetition is a draw condition in chess. The network needs to
        know how many times the current position has occurred to evaluate
        positions near a draw by repetition correctly.

**Auxiliary planes (7 planes)**

    112: Color plane            (all 1s if white to move, all 0s if black)
    113: Total move count       (all squares set to fullmove_number / 200.0)
    114: White kingside castle  (all 1s if right exists, all 0s otherwise)
    115: White queenside castle
    116: Black kingside castle
    117: Black queenside castle
    118: No-progress count      (all squares set to halfmove_clock / 100.0)

    Why normalize move counts?
        Neural networks train better when inputs are in a similar range.
        Dividing by 200 (a very long game) and 100 (the 50-move rule limit)
        keeps values roughly in [0, 1].

    Why full planes for scalar values?
        The AlphaZero paper encodes scalar values (like castling rights) as
        full 8x8 planes of constant values. This is redundant but makes the
        encoding uniform -- every feature is a spatial plane, and the
        convolutional layers can process them all identically.

Board Orientation
-----------------

When it is black's turn, the board is flipped vertically (rank 0 <-> rank 7).
This means:
    - The player to move always has their pieces on ranks 0-1 (from their
      perspective), just as white does in the standard starting position.
    - The network sees a "canonical" view regardless of whose turn it is.

Usage
-----

    >>> from neural.encoding import BoardState, encode_board, encode_batch
    >>> # Create a starting position
    >>> state = BoardState.initial()
    >>> tensor = encode_board(state)
    >>> tensor.shape
    torch.Size([119, 8, 8])
    >>> # Batch encoding
    >>> batch = encode_batch([state, state])
    >>> batch.shape
    torch.Size([2, 119, 8, 8])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import torch

from neural.config import NetworkConfig

# =============================================================================
# Constants: Piece Types
# =============================================================================
# We use IntEnum so that piece types can double as indices into the piece
# plane arrays. The ordering matches the AlphaZero paper convention.


class PieceType(IntEnum):
    """Chess piece types, ordered to match plane layout.

    The integer values serve as offsets within a time step's piece planes:
        - Planes 0-5: current player's pieces (indexed by PieceType value)
        - Planes 6-11: opponent's pieces (indexed by PieceType value + 6)
    """

    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5


class Color(IntEnum):
    """Side to move."""

    WHITE = 0
    BLACK = 1


# =============================================================================
# Constants: Plane Indices
# =============================================================================
# These constants make the plane layout explicit and avoid magic numbers
# throughout the encoding code.

# --- History planes (per time step) ---
PLANES_PER_TIME_STEP: int = 14
"""Each history time step has 12 piece planes + 2 repetition planes."""

HISTORY_STEPS: int = 8
"""Number of past positions to include (T=8 in the AlphaZero paper)."""

TOTAL_HISTORY_PLANES: int = PLANES_PER_TIME_STEP * HISTORY_STEPS
"""112 planes for the full history stack."""

# Piece plane offsets within a single time step
# Current player's pieces: offsets 0-5
CURRENT_PLAYER_PAWN_OFFSET: int = PieceType.PAWN        # 0
CURRENT_PLAYER_KNIGHT_OFFSET: int = PieceType.KNIGHT     # 1
CURRENT_PLAYER_BISHOP_OFFSET: int = PieceType.BISHOP     # 2
CURRENT_PLAYER_ROOK_OFFSET: int = PieceType.ROOK         # 3
CURRENT_PLAYER_QUEEN_OFFSET: int = PieceType.QUEEN       # 4
CURRENT_PLAYER_KING_OFFSET: int = PieceType.KING          # 5

# Opponent's pieces: offsets 6-11
OPPONENT_PIECE_OFFSET: int = 6
"""Add this to a PieceType value to get the opponent's plane offset."""

OPPONENT_PAWN_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.PAWN       # 6
OPPONENT_KNIGHT_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.KNIGHT   # 7
OPPONENT_BISHOP_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.BISHOP   # 8
OPPONENT_ROOK_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.ROOK       # 9
OPPONENT_QUEEN_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.QUEEN     # 10
OPPONENT_KING_OFFSET: int = OPPONENT_PIECE_OFFSET + PieceType.KING       # 11

# Repetition planes: offsets 12-13
REPETITION_1_OFFSET: int = 12
"""Plane is all 1s if position has occurred at least once before."""

REPETITION_2_OFFSET: int = 13
"""Plane is all 1s if position has occurred at least twice before."""

# --- Auxiliary planes (absolute indices) ---
COLOR_PLANE: int = TOTAL_HISTORY_PLANES          # 112
"""All 1s if white to move, all 0s if black to move."""

MOVE_COUNT_PLANE: int = TOTAL_HISTORY_PLANES + 1  # 113
"""Fullmove number divided by 200 (normalization constant)."""

CASTLING_WK_PLANE: int = TOTAL_HISTORY_PLANES + 2  # 114
"""White kingside castling right (all 1s if available)."""

CASTLING_WQ_PLANE: int = TOTAL_HISTORY_PLANES + 3  # 115
"""White queenside castling right (all 1s if available)."""

CASTLING_BK_PLANE: int = TOTAL_HISTORY_PLANES + 4  # 116
"""Black kingside castling right (all 1s if available)."""

CASTLING_BQ_PLANE: int = TOTAL_HISTORY_PLANES + 5  # 117
"""Black queenside castling right (all 1s if available)."""

NO_PROGRESS_PLANE: int = TOTAL_HISTORY_PLANES + 6  # 118
"""Halfmove clock divided by 100 (the 50-move rule limit)."""

TOTAL_PLANES: int = TOTAL_HISTORY_PLANES + 7      # 119
"""Total number of input planes. Must match NetworkConfig.input_planes."""

# Normalization constants
MOVE_COUNT_NORMALIZATION: float = 200.0
"""Divisor for fullmove number. 200 full moves is an extremely long game."""

HALFMOVE_CLOCK_NORMALIZATION: float = 100.0
"""Divisor for halfmove clock. 100 half-moves = 50-move rule threshold."""

# Board dimensions
BOARD_SIZE: int = 8
"""Chess board is 8x8."""


# =============================================================================
# Board State Representation
# =============================================================================


@dataclass
class CastlingRights:
    """Tracks which castling moves are still legally available.

    A castling right is lost when the king or the relevant rook moves,
    even if they later return to their original squares.
    """

    white_kingside: bool = True
    white_queenside: bool = True
    black_kingside: bool = True
    black_queenside: bool = True


# Type alias for the piece map: maps (rank, file) to (color, piece_type).
# rank 0 = rank 1 in standard notation (white's back rank).
# file 0 = a-file.
PieceMap = Dict[Tuple[int, int], Tuple[Color, PieceType]]


@dataclass
class BoardState:
    """All information needed to encode a chess position for the neural network.

    This is a pure Python representation of a chess position, designed for
    clarity rather than performance. When the Rust chess engine is connected
    via PyO3, we'll create a thin adapter that converts Rust board structs
    into this format.

    Coordinate system:
        - rank: 0-7, where 0 = rank 1 (white's back rank) and 7 = rank 8
        - file: 0-7, where 0 = a-file and 7 = h-file

    Attributes:
        pieces: Dictionary mapping (rank, file) -> (Color, PieceType).
            Only occupied squares are included. This is a sparse representation
            that's easy to construct and inspect.
        side_to_move: Which player moves next.
        castling: Current castling rights for both players.
        en_passant_square: The en passant target square as (rank, file), or
            None if no en passant capture is available. This is the square
            that a capturing pawn would move TO (rank 2 for white's en passant
            capture, rank 5 for black's).
        halfmove_clock: Number of half-moves since the last pawn advance or
            capture. Used for the 50-move draw rule (draw at 100 half-moves).
        fullmove_number: The full move number, starting at 1 and incremented
            after black's move. Used for normalization in the input encoding.
        repetition_count: How many times this exact position has occurred
            before in the current game (0 = first occurrence, 1 = second, etc.).
            Used for the threefold repetition rule and for the repetition
            input planes.
        history: List of previous board states, most recent first. The current
            position is NOT in this list -- it IS this BoardState. history[0]
            is the position one move ago, history[1] is two moves ago, etc.
            At most HISTORY_STEPS - 1 entries are used (7 past + current = 8).
    """

    pieces: PieceMap = field(default_factory=dict)
    side_to_move: Color = Color.WHITE
    castling: CastlingRights = field(default_factory=CastlingRights)
    en_passant_square: Optional[Tuple[int, int]] = None
    halfmove_clock: int = 0
    fullmove_number: int = 1
    repetition_count: int = 0
    history: List[BoardState] = field(default_factory=list)

    # -----------------------------------------------------------------
    # Factory methods
    # -----------------------------------------------------------------

    @classmethod
    def initial(cls) -> BoardState:
        """Create the standard chess starting position.

        Returns a BoardState with all pieces in their initial squares,
        white to move, all castling rights available, no en passant,
        and empty history.
        """
        pieces: PieceMap = {}

        # White pieces (rank 0 = rank 1 in standard notation)
        back_rank_order = [
            PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP,
            PieceType.QUEEN, PieceType.KING, PieceType.BISHOP,
            PieceType.KNIGHT, PieceType.ROOK,
        ]
        for file_idx, piece_type in enumerate(back_rank_order):
            pieces[(0, file_idx)] = (Color.WHITE, piece_type)
        for file_idx in range(BOARD_SIZE):
            pieces[(1, file_idx)] = (Color.WHITE, PieceType.PAWN)

        # Black pieces (rank 7 = rank 8 in standard notation)
        for file_idx, piece_type in enumerate(back_rank_order):
            pieces[(7, file_idx)] = (Color.BLACK, piece_type)
        for file_idx in range(BOARD_SIZE):
            pieces[(6, file_idx)] = (Color.BLACK, PieceType.PAWN)

        return cls(
            pieces=pieces,
            side_to_move=Color.WHITE,
            castling=CastlingRights(),
            en_passant_square=None,
            halfmove_clock=0,
            fullmove_number=1,
            repetition_count=0,
            history=[],
        )

    @classmethod
    def empty(cls) -> BoardState:
        """Create an empty board with no pieces.

        Useful for testing specific piece configurations.
        """
        return cls(
            pieces={},
            side_to_move=Color.WHITE,
            castling=CastlingRights(
                white_kingside=False,
                white_queenside=False,
                black_kingside=False,
                black_queenside=False,
            ),
            en_passant_square=None,
            halfmove_clock=0,
            fullmove_number=1,
            repetition_count=0,
            history=[],
        )

    @classmethod
    def from_fen_piece_placement(
        cls,
        fen: str,
        side_to_move: Color = Color.WHITE,
        castling: Optional[CastlingRights] = None,
        en_passant_square: Optional[Tuple[int, int]] = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
        repetition_count: int = 0,
        history: Optional[List[BoardState]] = None,
    ) -> BoardState:
        """Create a BoardState from a FEN string.

        Parses a complete FEN string (or just the piece-placement part) and
        constructs the corresponding BoardState.

        Args:
            fen: A FEN string. Can be just the piece placement field
                (e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR") or
                a full FEN with all six fields.
            side_to_move: Override side to move (used if FEN has only piece
                placement). If full FEN is provided, this is ignored.
            castling: Override castling rights. If None and full FEN is
                provided, parsed from FEN. If None and only piece placement,
                defaults to no castling rights.
            en_passant_square: Override en passant square.
            halfmove_clock: Override halfmove clock.
            fullmove_number: Override fullmove number.
            repetition_count: Repetition count for this position.
            history: List of previous BoardStates.

        Returns:
            A new BoardState.
        """
        fen_parts = fen.strip().split()
        piece_placement = fen_parts[0]

        # Parse additional FEN fields if present
        if len(fen_parts) >= 2:
            side_to_move = Color.WHITE if fen_parts[1] == "w" else Color.BLACK
        if len(fen_parts) >= 3 and castling is None:
            castling_str = fen_parts[2]
            castling = CastlingRights(
                white_kingside="K" in castling_str,
                white_queenside="Q" in castling_str,
                black_kingside="k" in castling_str,
                black_queenside="q" in castling_str,
            )
        if len(fen_parts) >= 4 and fen_parts[3] != "-" and en_passant_square is None:
            ep_file = ord(fen_parts[3][0]) - ord("a")
            ep_rank = int(fen_parts[3][1]) - 1
            en_passant_square = (ep_rank, ep_file)
        if len(fen_parts) >= 5:
            halfmove_clock = int(fen_parts[4])
        if len(fen_parts) >= 6:
            fullmove_number = int(fen_parts[5])

        if castling is None:
            castling = CastlingRights(
                white_kingside=False,
                white_queenside=False,
                black_kingside=False,
                black_queenside=False,
            )

        # Parse piece placement (FEN ranks go from rank 8 down to rank 1)
        fen_piece_map = {
            "P": (Color.WHITE, PieceType.PAWN),
            "N": (Color.WHITE, PieceType.KNIGHT),
            "B": (Color.WHITE, PieceType.BISHOP),
            "R": (Color.WHITE, PieceType.ROOK),
            "Q": (Color.WHITE, PieceType.QUEEN),
            "K": (Color.WHITE, PieceType.KING),
            "p": (Color.BLACK, PieceType.PAWN),
            "n": (Color.BLACK, PieceType.KNIGHT),
            "b": (Color.BLACK, PieceType.BISHOP),
            "r": (Color.BLACK, PieceType.ROOK),
            "q": (Color.BLACK, PieceType.QUEEN),
            "k": (Color.BLACK, PieceType.KING),
        }

        pieces: PieceMap = {}
        ranks = piece_placement.split("/")
        for rank_from_top, rank_str in enumerate(ranks):
            rank = 7 - rank_from_top  # FEN starts from rank 8 (index 7)
            file_idx = 0
            for char in rank_str:
                if char.isdigit():
                    file_idx += int(char)
                else:
                    pieces[(rank, file_idx)] = fen_piece_map[char]
                    file_idx += 1

        return cls(
            pieces=pieces,
            side_to_move=side_to_move,
            castling=castling,
            en_passant_square=en_passant_square,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
            repetition_count=repetition_count,
            history=history or [],
        )


# =============================================================================
# Encoding Functions
# =============================================================================


def _encode_piece_planes(
    pieces: PieceMap,
    side_to_move: Color,
    flip: bool,
) -> torch.Tensor:
    """Encode piece positions into 12 binary planes (6 per player).

    The encoding is always from the perspective of the current player:
    planes 0-5 hold the current player's pieces, and planes 6-11 hold the
    opponent's pieces. When it's black's turn, the board is flipped vertically
    so black's pieces appear on the "white side" (low ranks).

    Args:
        pieces: The piece map for this position.
        side_to_move: Who is to move in THIS position (determines perspective).
        flip: Whether to flip the board vertically (True when the original
            position has black to move).

    Returns:
        Tensor of shape (12, 8, 8) with binary values.
    """
    planes = torch.zeros(12, BOARD_SIZE, BOARD_SIZE)

    for (rank, file_idx), (color, piece_type) in pieces.items():
        # Determine the output rank, flipping if needed
        out_rank = (7 - rank) if flip else rank

        # Determine if this piece belongs to the current player or opponent
        if color == side_to_move:
            plane_idx = int(piece_type)  # 0-5 for current player
        else:
            plane_idx = OPPONENT_PIECE_OFFSET + int(piece_type)  # 6-11 for opponent

        planes[plane_idx, out_rank, file_idx] = 1.0

    return planes


def _encode_repetition_planes(repetition_count: int) -> torch.Tensor:
    """Encode repetition count as 2 binary planes.

    Plane 0: all 1s if position has occurred >= 1 time before (repetition_count >= 1)
    Plane 1: all 1s if position has occurred >= 2 times before (repetition_count >= 2)

    Args:
        repetition_count: How many times this position has occurred before.

    Returns:
        Tensor of shape (2, 8, 8).
    """
    planes = torch.zeros(2, BOARD_SIZE, BOARD_SIZE)
    if repetition_count >= 1:
        planes[0] = 1.0
    if repetition_count >= 2:
        planes[1] = 1.0
    return planes


def _encode_single_time_step(
    state: BoardState,
    flip: bool,
) -> torch.Tensor:
    """Encode one time step (14 planes: 12 piece + 2 repetition).

    Args:
        state: The board state for this time step.
        flip: Whether to flip the board (True when the CURRENT player is black,
            meaning all historical positions also get flipped).

    Returns:
        Tensor of shape (14, 8, 8).
    """
    piece_planes = _encode_piece_planes(
        state.pieces, state.side_to_move, flip
    )
    repetition_planes = _encode_repetition_planes(state.repetition_count)
    return torch.cat([piece_planes, repetition_planes], dim=0)


def _encode_auxiliary_planes(state: BoardState) -> torch.Tensor:
    """Encode the 7 auxiliary feature planes.

    These planes capture game-level information that doesn't vary by position
    in the history:
        - Side to move
        - Move count (normalized)
        - Castling rights (4 planes)
        - No-progress count (normalized)

    Args:
        state: The current board state (not a historical one).

    Returns:
        Tensor of shape (7, 8, 8).
    """
    planes = torch.zeros(7, BOARD_SIZE, BOARD_SIZE)

    # Plane 0 (index 112 overall): Color -- all 1s if white to move
    if state.side_to_move == Color.WHITE:
        planes[0] = 1.0

    # Plane 1 (index 113 overall): Total move count, normalized
    planes[1] = state.fullmove_number / MOVE_COUNT_NORMALIZATION

    # Planes 2-5 (indices 114-117 overall): Castling rights
    if state.castling.white_kingside:
        planes[2] = 1.0
    if state.castling.white_queenside:
        planes[3] = 1.0
    if state.castling.black_kingside:
        planes[4] = 1.0
    if state.castling.black_queenside:
        planes[5] = 1.0

    # Plane 6 (index 118 overall): No-progress count (halfmove clock), normalized
    planes[6] = state.halfmove_clock / HALFMOVE_CLOCK_NORMALIZATION

    return planes


def encode_board(state: BoardState) -> torch.Tensor:
    """Encode a single chess position into a neural network input tensor.

    This is the main encoding function. It produces a tensor of shape
    (TOTAL_PLANES, 8, 8) = (119, 8, 8) that can be fed directly to the
    AlphaZero neural network.

    The encoding includes:
    - T=8 time steps of piece positions and repetition counts (112 planes)
    - 7 auxiliary planes (color, move count, castling, no-progress count)

    If fewer than T=8 historical positions are available (e.g., at the start
    of a game), the missing time steps are filled with zeros. This is the
    standard approach -- the network learns that zero planes mean "no
    information available for this time step."

    Board orientation: when black is to move, the board is flipped vertically
    so the network always sees the position from the current player's perspective.

    Args:
        state: The current board state to encode, including history.

    Returns:
        Tensor of shape (119, 8, 8) with dtype float32.
    """
    # Determine if we need to flip the board
    # We flip when the current player (who is about to move) is black
    flip = (state.side_to_move == Color.BLACK)

    # Allocate the full tensor
    result = torch.zeros(TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)

    # --- Encode current position (time step 0) ---
    time_step_planes = _encode_single_time_step(state, flip)
    result[0:PLANES_PER_TIME_STEP] = time_step_planes

    # --- Encode historical positions (time steps 1 through T-1) ---
    # history[0] is the most recent past position (1 move ago)
    num_history = min(len(state.history), HISTORY_STEPS - 1)
    for t in range(num_history):
        past_state = state.history[t]
        start_plane = (t + 1) * PLANES_PER_TIME_STEP
        end_plane = start_plane + PLANES_PER_TIME_STEP
        time_step_planes = _encode_single_time_step(past_state, flip)
        result[start_plane:end_plane] = time_step_planes

    # --- Encode auxiliary planes ---
    auxiliary = _encode_auxiliary_planes(state)
    result[TOTAL_HISTORY_PLANES:TOTAL_PLANES] = auxiliary

    return result


def encode_batch(states: List[BoardState]) -> torch.Tensor:
    """Encode multiple board positions into a batched tensor.

    This is a convenience function that encodes each position individually
    and stacks them into a batch. For training, this produces the input
    tensor expected by the network's forward pass.

    Args:
        states: List of board states to encode.

    Returns:
        Tensor of shape (B, 119, 8, 8) where B = len(states).

    Raises:
        ValueError: If the states list is empty.
    """
    if not states:
        raise ValueError("Cannot encode an empty batch of states.")

    tensors = [encode_board(state) for state in states]
    return torch.stack(tensors, dim=0)
