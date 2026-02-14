"""
Tests for Board Encoding
========================

These tests verify that chess positions are correctly converted into neural
network input tensors. We test the encoding at multiple levels:

1. Constants: verify plane indices and counts are internally consistent.
2. BoardState: verify factory methods produce correct piece layouts.
3. Piece planes: verify individual piece types are encoded on the right planes.
4. Repetition planes: verify the binary repetition encoding.
5. Auxiliary planes: verify color, castling, move count, and no-progress planes.
6. Board flipping: verify the board is flipped when black is to move.
7. History stacking: verify that multiple time steps are encoded correctly.
8. Batch encoding: verify that batch encoding produces correct shapes.
9. Integration: encode known positions and check full tensor properties.
"""

import torch
import pytest

from neural.encoding import (
    # Constants
    TOTAL_PLANES,
    TOTAL_HISTORY_PLANES,
    PLANES_PER_TIME_STEP,
    HISTORY_STEPS,
    BOARD_SIZE,
    OPPONENT_PIECE_OFFSET,
    REPETITION_1_OFFSET,
    REPETITION_2_OFFSET,
    COLOR_PLANE,
    MOVE_COUNT_PLANE,
    CASTLING_WK_PLANE,
    CASTLING_WQ_PLANE,
    CASTLING_BK_PLANE,
    CASTLING_BQ_PLANE,
    NO_PROGRESS_PLANE,
    MOVE_COUNT_NORMALIZATION,
    HALFMOVE_CLOCK_NORMALIZATION,
    # Types
    PieceType,
    Color,
    CastlingRights,
    BoardState,
    # Functions
    encode_board,
    encode_batch,
    _encode_piece_planes,
    _encode_repetition_planes,
    _encode_auxiliary_planes,
)

from neural.config import NetworkConfig


# =============================================================================
# 1. Constant Consistency
# =============================================================================


class TestConstants:
    """Verify that our plane layout constants are internally consistent."""

    def test_total_planes_matches_config(self) -> None:
        """TOTAL_PLANES must match the default NetworkConfig.input_planes."""
        config = NetworkConfig()
        assert TOTAL_PLANES == config.input_planes, (
            f"TOTAL_PLANES ({TOTAL_PLANES}) does not match "
            f"NetworkConfig.input_planes ({config.input_planes}). "
            f"The encoding and network must agree on input size."
        )

    def test_total_planes_is_119(self) -> None:
        """The AlphaZero paper specifies 119 input planes."""
        assert TOTAL_PLANES == 119

    def test_history_planes_decomposition(self) -> None:
        """112 history planes = 8 steps x 14 planes per step."""
        assert TOTAL_HISTORY_PLANES == HISTORY_STEPS * PLANES_PER_TIME_STEP
        assert TOTAL_HISTORY_PLANES == 112

    def test_planes_per_time_step(self) -> None:
        """Each time step: 12 piece planes + 2 repetition planes = 14."""
        assert PLANES_PER_TIME_STEP == 14

    def test_auxiliary_planes_count(self) -> None:
        """7 auxiliary planes: color + move count + 4 castling + no-progress."""
        auxiliary_count = TOTAL_PLANES - TOTAL_HISTORY_PLANES
        assert auxiliary_count == 7

    def test_auxiliary_plane_indices(self) -> None:
        """Verify each auxiliary plane has the correct absolute index."""
        assert COLOR_PLANE == 112
        assert MOVE_COUNT_PLANE == 113
        assert CASTLING_WK_PLANE == 114
        assert CASTLING_WQ_PLANE == 115
        assert CASTLING_BK_PLANE == 116
        assert CASTLING_BQ_PLANE == 117
        assert NO_PROGRESS_PLANE == 118

    def test_piece_type_values(self) -> None:
        """PieceType enum values serve as plane offsets."""
        assert PieceType.PAWN == 0
        assert PieceType.KNIGHT == 1
        assert PieceType.BISHOP == 2
        assert PieceType.ROOK == 3
        assert PieceType.QUEEN == 4
        assert PieceType.KING == 5

    def test_opponent_offset(self) -> None:
        """Opponent's piece planes start 6 after the current player's."""
        assert OPPONENT_PIECE_OFFSET == 6


# =============================================================================
# 2. BoardState Factory Methods
# =============================================================================


class TestBoardState:
    """Verify BoardState construction."""

    def test_initial_position_piece_count(self) -> None:
        """Starting position has 32 pieces."""
        state = BoardState.initial()
        assert len(state.pieces) == 32

    def test_initial_position_white_pieces(self) -> None:
        """White has 16 pieces in the starting position."""
        state = BoardState.initial()
        white_pieces = [
            (pos, pt)
            for pos, (color, pt) in state.pieces.items()
            if color == Color.WHITE
        ]
        assert len(white_pieces) == 16

    def test_initial_position_black_pieces(self) -> None:
        """Black has 16 pieces in the starting position."""
        state = BoardState.initial()
        black_pieces = [
            (pos, pt)
            for pos, (color, pt) in state.pieces.items()
            if color == Color.BLACK
        ]
        assert len(black_pieces) == 16

    def test_initial_position_white_pawns(self) -> None:
        """White pawns are on rank 1 (index 1)."""
        state = BoardState.initial()
        for file_idx in range(8):
            color, piece_type = state.pieces[(1, file_idx)]
            assert color == Color.WHITE
            assert piece_type == PieceType.PAWN

    def test_initial_position_black_pawns(self) -> None:
        """Black pawns are on rank 6 (index 6)."""
        state = BoardState.initial()
        for file_idx in range(8):
            color, piece_type = state.pieces[(6, file_idx)]
            assert color == Color.BLACK
            assert piece_type == PieceType.PAWN

    def test_initial_position_kings(self) -> None:
        """Kings are on the e-file (index 4)."""
        state = BoardState.initial()
        assert state.pieces[(0, 4)] == (Color.WHITE, PieceType.KING)
        assert state.pieces[(7, 4)] == (Color.BLACK, PieceType.KING)

    def test_initial_position_queens(self) -> None:
        """Queens are on the d-file (index 3)."""
        state = BoardState.initial()
        assert state.pieces[(0, 3)] == (Color.WHITE, PieceType.QUEEN)
        assert state.pieces[(7, 3)] == (Color.BLACK, PieceType.QUEEN)

    def test_initial_position_defaults(self) -> None:
        """Verify all non-piece fields in the starting position."""
        state = BoardState.initial()
        assert state.side_to_move == Color.WHITE
        assert state.castling.white_kingside is True
        assert state.castling.white_queenside is True
        assert state.castling.black_kingside is True
        assert state.castling.black_queenside is True
        assert state.en_passant_square is None
        assert state.halfmove_clock == 0
        assert state.fullmove_number == 1
        assert state.repetition_count == 0
        assert state.history == []

    def test_empty_board(self) -> None:
        """Empty board has no pieces and no castling rights."""
        state = BoardState.empty()
        assert len(state.pieces) == 0
        assert state.castling.white_kingside is False
        assert state.castling.white_queenside is False
        assert state.castling.black_kingside is False
        assert state.castling.black_queenside is False

    def test_from_fen_starting_position(self) -> None:
        """Parse the starting position FEN."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        state = BoardState.from_fen_piece_placement(fen)
        initial = BoardState.initial()
        assert state.pieces == initial.pieces
        assert state.side_to_move == Color.WHITE
        assert state.castling.white_kingside is True
        assert state.castling.white_queenside is True
        assert state.castling.black_kingside is True
        assert state.castling.black_queenside is True

    def test_from_fen_mid_game(self) -> None:
        """Parse a mid-game FEN with specific features."""
        # After 1. e4 e5 2. Nf3 (Italian-ish)
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
        state = BoardState.from_fen_piece_placement(fen)
        assert state.side_to_move == Color.BLACK
        assert state.halfmove_clock == 1
        assert state.fullmove_number == 2
        # White knight on f3 (rank 2, file 5)
        assert state.pieces[(2, 5)] == (Color.WHITE, PieceType.KNIGHT)
        # Black pawn on e5 (rank 4, file 4)
        assert state.pieces[(4, 4)] == (Color.BLACK, PieceType.PAWN)

    def test_from_fen_en_passant(self) -> None:
        """Parse a FEN with an en passant square."""
        fen = "rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3"
        state = BoardState.from_fen_piece_placement(fen)
        assert state.en_passant_square == (5, 4)  # e6 = rank 5, file 4

    def test_from_fen_no_castling(self) -> None:
        """Parse a FEN with no castling rights."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"
        state = BoardState.from_fen_piece_placement(fen)
        assert state.castling.white_kingside is False
        assert state.castling.white_queenside is False
        assert state.castling.black_kingside is False
        assert state.castling.black_queenside is False

    def test_from_fen_partial_castling(self) -> None:
        """Parse a FEN with partial castling rights."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w Kq - 0 1"
        state = BoardState.from_fen_piece_placement(fen)
        assert state.castling.white_kingside is True
        assert state.castling.white_queenside is False
        assert state.castling.black_kingside is False
        assert state.castling.black_queenside is True


# =============================================================================
# 3. Piece Plane Encoding
# =============================================================================


class TestPiecePlanes:
    """Verify that pieces are placed on the correct planes and squares."""

    def test_white_pawn_plane(self) -> None:
        """White pawn on e2 (rank 1, file 4) when white to move."""
        state = BoardState.empty()
        state.pieces[(1, 4)] = (Color.WHITE, PieceType.PAWN)
        state.side_to_move = Color.WHITE

        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=False)

        # Pawn is on plane 0 (current player's pawn)
        assert planes[PieceType.PAWN, 1, 4] == 1.0
        # All other squares on this plane are 0
        assert planes[PieceType.PAWN].sum() == 1.0
        # All other planes are empty
        for p in range(12):
            if p != PieceType.PAWN:
                assert planes[p].sum() == 0.0

    def test_opponent_piece_offset(self) -> None:
        """Black knight on b8 appears on opponent's knight plane when white moves."""
        state = BoardState.empty()
        state.pieces[(7, 1)] = (Color.BLACK, PieceType.KNIGHT)
        state.side_to_move = Color.WHITE

        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=False)

        # Black knight from white's perspective is on opponent plane
        opp_knight = OPPONENT_PIECE_OFFSET + PieceType.KNIGHT
        assert planes[opp_knight, 7, 1] == 1.0
        assert planes[opp_knight].sum() == 1.0

    def test_starting_position_piece_counts(self) -> None:
        """Starting position: 8 pawns per side, 2 knights/bishops/rooks, 1 queen, 1 king."""
        state = BoardState.initial()
        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=False)

        # Current player (white) pieces
        assert planes[PieceType.PAWN].sum() == 8
        assert planes[PieceType.KNIGHT].sum() == 2
        assert planes[PieceType.BISHOP].sum() == 2
        assert planes[PieceType.ROOK].sum() == 2
        assert planes[PieceType.QUEEN].sum() == 1
        assert planes[PieceType.KING].sum() == 1

        # Opponent (black) pieces
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.PAWN].sum() == 8
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.KNIGHT].sum() == 2
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.BISHOP].sum() == 2
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.ROOK].sum() == 2
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.QUEEN].sum() == 1
        assert planes[OPPONENT_PIECE_OFFSET + PieceType.KING].sum() == 1

    def test_total_pieces_starting_position(self) -> None:
        """All 12 planes together have exactly 32 non-zero entries."""
        state = BoardState.initial()
        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=False)
        assert planes.sum() == 32.0

    def test_no_overlap_between_planes(self) -> None:
        """No two planes should have a 1 on the same square (each square has at most one piece)."""
        state = BoardState.initial()
        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=False)
        # Sum across all 12 planes per square: should be 0 or 1
        occupancy = planes.sum(dim=0)
        assert occupancy.max() <= 1.0


# =============================================================================
# 4. Board Flipping (Black to Move)
# =============================================================================


class TestBoardFlipping:
    """Verify the board is flipped vertically when black is to move."""

    def test_flip_white_pawn(self) -> None:
        """A white pawn on rank 1 should appear on rank 6 after flipping."""
        state = BoardState.empty()
        state.pieces[(1, 4)] = (Color.WHITE, PieceType.PAWN)
        state.side_to_move = Color.BLACK  # Black to move

        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=True)

        # White is the opponent when black moves -> opponent pawn plane
        opp_pawn = OPPONENT_PIECE_OFFSET + PieceType.PAWN
        # Rank 1 flipped = rank 6
        assert planes[opp_pawn, 6, 4] == 1.0
        assert planes[opp_pawn].sum() == 1.0

    def test_flip_black_pawn(self) -> None:
        """A black pawn on rank 6 should appear on rank 1 after flipping (current player's perspective)."""
        state = BoardState.empty()
        state.pieces[(6, 4)] = (Color.BLACK, PieceType.PAWN)
        state.side_to_move = Color.BLACK

        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=True)

        # Black is current player when black moves -> current player's pawn plane
        # Rank 6 flipped = rank 1
        assert planes[PieceType.PAWN, 1, 4] == 1.0

    def test_starting_position_flipped(self) -> None:
        """When black moves from starting position, the board is mirrored.

        Black's back rank (rank 7) should appear on rank 0 after flipping,
        and black's pieces should be on the current player's planes.
        """
        state = BoardState.initial()
        state.side_to_move = Color.BLACK

        planes = _encode_piece_planes(state.pieces, state.side_to_move, flip=True)

        # Black's king was on (7, 4), after flip it's on (0, 4), current player's king plane
        assert planes[PieceType.KING, 0, 4] == 1.0

        # White's king was on (0, 4), after flip it's on (7, 4), opponent's king plane
        opp_king = OPPONENT_PIECE_OFFSET + PieceType.KING
        assert planes[opp_king, 7, 4] == 1.0

        # Black's pawns were on rank 6, after flip they're on rank 1
        for f in range(8):
            assert planes[PieceType.PAWN, 1, f] == 1.0

        # White's pawns were on rank 1, after flip they're on rank 6
        opp_pawn = OPPONENT_PIECE_OFFSET + PieceType.PAWN
        for f in range(8):
            assert planes[opp_pawn, 6, f] == 1.0


# =============================================================================
# 5. Repetition Planes
# =============================================================================


class TestRepetitionPlanes:
    """Verify repetition count encoding."""

    def test_no_repetition(self) -> None:
        """repetition_count=0: both planes are all zeros."""
        planes = _encode_repetition_planes(0)
        assert planes[0].sum() == 0.0
        assert planes[1].sum() == 0.0

    def test_one_repetition(self) -> None:
        """repetition_count=1: first plane all ones, second all zeros."""
        planes = _encode_repetition_planes(1)
        assert planes[0].sum() == 64.0  # 8x8 = 64 ones
        assert planes[1].sum() == 0.0

    def test_two_repetitions(self) -> None:
        """repetition_count=2: both planes all ones."""
        planes = _encode_repetition_planes(2)
        assert planes[0].sum() == 64.0
        assert planes[1].sum() == 64.0

    def test_high_repetition(self) -> None:
        """repetition_count > 2: same as 2 (both planes saturate)."""
        planes = _encode_repetition_planes(5)
        assert planes[0].sum() == 64.0
        assert planes[1].sum() == 64.0

    def test_shape(self) -> None:
        """Repetition planes have shape (2, 8, 8)."""
        planes = _encode_repetition_planes(0)
        assert planes.shape == (2, 8, 8)


# =============================================================================
# 6. Auxiliary Planes
# =============================================================================


class TestAuxiliaryPlanes:
    """Verify the 7 auxiliary feature planes."""

    def test_color_plane_white(self) -> None:
        """Color plane is all 1s when white to move."""
        state = BoardState.initial()
        aux = _encode_auxiliary_planes(state)
        assert aux[0].sum() == 64.0  # All 8x8 = 64 squares are 1

    def test_color_plane_black(self) -> None:
        """Color plane is all 0s when black to move."""
        state = BoardState.initial()
        state.side_to_move = Color.BLACK
        aux = _encode_auxiliary_planes(state)
        assert aux[0].sum() == 0.0

    def test_move_count_plane(self) -> None:
        """Move count plane has uniform value = fullmove_number / 200."""
        state = BoardState.empty()
        state.fullmove_number = 50
        aux = _encode_auxiliary_planes(state)
        expected = 50.0 / 200.0
        assert torch.allclose(aux[1], torch.full((8, 8), expected))

    def test_move_count_plane_start(self) -> None:
        """At the start of the game, move count plane = 1/200."""
        state = BoardState.initial()
        aux = _encode_auxiliary_planes(state)
        expected = 1.0 / 200.0
        assert torch.allclose(aux[1], torch.full((8, 8), expected))

    def test_castling_planes_all_rights(self) -> None:
        """All castling rights -> all four castling planes are all 1s."""
        state = BoardState.initial()
        aux = _encode_auxiliary_planes(state)
        for i in range(2, 6):  # planes 2-5 = four castling planes
            assert aux[i].sum() == 64.0, f"Castling plane {i} should be all 1s"

    def test_castling_planes_no_rights(self) -> None:
        """No castling rights -> all four castling planes are all 0s."""
        state = BoardState.empty()
        aux = _encode_auxiliary_planes(state)
        for i in range(2, 6):
            assert aux[i].sum() == 0.0, f"Castling plane {i} should be all 0s"

    def test_castling_planes_partial(self) -> None:
        """Only white kingside castling -> only that plane is 1s."""
        state = BoardState.empty()
        state.castling = CastlingRights(
            white_kingside=True,
            white_queenside=False,
            black_kingside=False,
            black_queenside=False,
        )
        aux = _encode_auxiliary_planes(state)
        assert aux[2].sum() == 64.0   # WK
        assert aux[3].sum() == 0.0    # WQ
        assert aux[4].sum() == 0.0    # BK
        assert aux[5].sum() == 0.0    # BQ

    def test_no_progress_plane(self) -> None:
        """No-progress plane = halfmove_clock / 100."""
        state = BoardState.empty()
        state.halfmove_clock = 30
        aux = _encode_auxiliary_planes(state)
        expected = 30.0 / 100.0
        assert torch.allclose(aux[6], torch.full((8, 8), expected))

    def test_no_progress_plane_at_zero(self) -> None:
        """At game start, no-progress plane is all zeros."""
        state = BoardState.initial()
        aux = _encode_auxiliary_planes(state)
        assert aux[6].sum() == 0.0

    def test_auxiliary_shape(self) -> None:
        """Auxiliary planes have shape (7, 8, 8)."""
        state = BoardState.initial()
        aux = _encode_auxiliary_planes(state)
        assert aux.shape == (7, 8, 8)


# =============================================================================
# 7. Full Board Encoding (encode_board)
# =============================================================================


class TestEncodeBoard:
    """Test the main encode_board function end-to-end."""

    def test_output_shape(self) -> None:
        """Output tensor has shape (119, 8, 8)."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor.shape == (TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)

    def test_output_dtype(self) -> None:
        """Output tensor is float32."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor.dtype == torch.float32

    def test_starting_position_current_player_pawns(self) -> None:
        """In the starting position (white to move), white pawns on rank 1."""
        state = BoardState.initial()
        tensor = encode_board(state)
        # Time step 0, current player's pawn plane = plane 0
        for f in range(8):
            assert tensor[PieceType.PAWN, 1, f] == 1.0
        # No pawns elsewhere on this plane
        assert tensor[PieceType.PAWN].sum() == 8.0

    def test_starting_position_opponent_pawns(self) -> None:
        """In the starting position, black pawns (opponent) on rank 6."""
        state = BoardState.initial()
        tensor = encode_board(state)
        opp_pawn = OPPONENT_PIECE_OFFSET + PieceType.PAWN
        for f in range(8):
            assert tensor[opp_pawn, 6, f] == 1.0
        assert tensor[opp_pawn].sum() == 8.0

    def test_starting_position_king_locations(self) -> None:
        """White king on (0,4) and black king on (7,4) in starting position."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor[PieceType.KING, 0, 4] == 1.0
        opp_king = OPPONENT_PIECE_OFFSET + PieceType.KING
        assert tensor[opp_king, 7, 4] == 1.0

    def test_empty_history_fills_zeros(self) -> None:
        """With no history, time steps 1-7 should be all zeros."""
        state = BoardState.initial()
        tensor = encode_board(state)
        for t in range(1, HISTORY_STEPS):
            start = t * PLANES_PER_TIME_STEP
            end = start + PLANES_PER_TIME_STEP
            assert tensor[start:end].sum() == 0.0, (
                f"Time step {t} (planes {start}-{end}) should be all zeros "
                f"when no history is available."
            )

    def test_color_plane_in_full_encoding(self) -> None:
        """Color plane (index 112) is all 1s for white to move."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor[COLOR_PLANE].sum() == 64.0

    def test_castling_planes_in_full_encoding(self) -> None:
        """All four castling planes are all 1s in starting position."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor[CASTLING_WK_PLANE].sum() == 64.0
        assert tensor[CASTLING_WQ_PLANE].sum() == 64.0
        assert tensor[CASTLING_BK_PLANE].sum() == 64.0
        assert tensor[CASTLING_BQ_PLANE].sum() == 64.0

    def test_no_progress_plane_starting_position(self) -> None:
        """Halfmove clock is 0 at start -> no-progress plane is all zeros."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert tensor[NO_PROGRESS_PLANE].sum() == 0.0

    def test_values_are_non_negative(self) -> None:
        """All values in the encoding should be non-negative."""
        state = BoardState.initial()
        tensor = encode_board(state)
        assert (tensor >= 0.0).all()

    def test_black_to_move_flips_board(self) -> None:
        """When black moves, the board should be flipped.

        Create a position with just a white king on e1 (rank 0, file 4)
        and encode from black's perspective. The white king (opponent)
        should appear on rank 7 after flipping.
        """
        state = BoardState.empty()
        state.pieces[(0, 4)] = (Color.WHITE, PieceType.KING)
        state.side_to_move = Color.BLACK

        tensor = encode_board(state)

        # White king is opponent from black's perspective
        opp_king = OPPONENT_PIECE_OFFSET + PieceType.KING
        # Rank 0 flips to rank 7
        assert tensor[opp_king, 7, 4] == 1.0


# =============================================================================
# 8. History Stacking
# =============================================================================


class TestHistoryStacking:
    """Verify that historical positions are encoded in the correct time steps."""

    def test_single_history_step(self) -> None:
        """One historical position should appear in time step 1."""
        # Current position: empty
        current = BoardState.empty()
        current.side_to_move = Color.WHITE

        # Previous position: white king on e1
        prev = BoardState.empty()
        prev.pieces[(0, 4)] = (Color.WHITE, PieceType.KING)
        prev.side_to_move = Color.BLACK  # It was black's turn before current

        current.history = [prev]

        tensor = encode_board(current)

        # Time step 0 (current) should have no pieces
        assert tensor[0:12].sum() == 0.0

        # Time step 1 should have the king from prev
        # The encoding always flips based on the CURRENT player's perspective
        # Current player is white (no flip), so the king from prev appears as-is
        # The king in prev belonged to white, and the prev state had side_to_move=BLACK
        # From current player (white) perspective: white is current player
        # But the encoding of a historical step uses the prev state's side_to_move
        # to determine current/opponent, and flips based on current state's side_to_move
        t1_start = PLANES_PER_TIME_STEP
        # White king from prev: prev.side_to_move was BLACK, so white is opponent
        # -> opponent king plane at offset 11. Flip=False (current is white).
        opp_king_in_t1 = t1_start + OPPONENT_PIECE_OFFSET + PieceType.KING
        assert tensor[opp_king_in_t1, 0, 4] == 1.0

    def test_full_history(self) -> None:
        """With 7 history entries, all 8 time steps should be populated."""
        current = BoardState.empty()
        current.side_to_move = Color.WHITE
        current.pieces[(0, 4)] = (Color.WHITE, PieceType.KING)

        # Create 7 historical states (each with a white king in different positions)
        history_states = []
        for i in range(7):
            past = BoardState.empty()
            past.side_to_move = Color.BLACK if (i % 2 == 0) else Color.WHITE
            past.pieces[(0, i)] = (Color.WHITE, PieceType.KING)
            history_states.append(past)

        current.history = history_states
        tensor = encode_board(current)

        # Time step 0 (current): white king at (0,4), current player's plane
        assert tensor[PieceType.KING, 0, 4] == 1.0

        # Each historical time step should have exactly one non-zero piece entry
        for t in range(1, HISTORY_STEPS):
            t_start = t * PLANES_PER_TIME_STEP
            piece_planes = tensor[t_start : t_start + 12]
            assert piece_planes.sum() == 1.0, (
                f"Time step {t} should have exactly one piece"
            )

    def test_excess_history_is_ignored(self) -> None:
        """If more than HISTORY_STEPS-1 entries are in history, extras are ignored."""
        current = BoardState.empty()
        current.side_to_move = Color.WHITE

        # 10 history entries (only 7 should be used)
        history_states = []
        for i in range(10):
            past = BoardState.empty()
            past.side_to_move = Color.BLACK
            past.pieces[(0, 0)] = (Color.WHITE, PieceType.PAWN)
            history_states.append(past)

        current.history = history_states
        tensor = encode_board(current)

        # The tensor shape should still be (119, 8, 8)
        assert tensor.shape == (TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)

    def test_repetition_in_history(self) -> None:
        """Historical positions with repetitions have correct repetition planes."""
        current = BoardState.empty()
        current.side_to_move = Color.WHITE
        current.repetition_count = 1  # Current position seen once before

        prev = BoardState.empty()
        prev.side_to_move = Color.BLACK
        prev.repetition_count = 0  # Previous position not repeated

        current.history = [prev]
        tensor = encode_board(current)

        # Time step 0: repetition plane 1 should be all 1s
        assert tensor[REPETITION_1_OFFSET].sum() == 64.0
        assert tensor[REPETITION_2_OFFSET].sum() == 0.0

        # Time step 1: repetition planes should be all 0s
        t1_rep1 = PLANES_PER_TIME_STEP + REPETITION_1_OFFSET
        t1_rep2 = PLANES_PER_TIME_STEP + REPETITION_2_OFFSET
        assert tensor[t1_rep1].sum() == 0.0
        assert tensor[t1_rep2].sum() == 0.0


# =============================================================================
# 9. Batch Encoding
# =============================================================================


class TestEncodeBatch:
    """Verify batch encoding for multiple positions."""

    def test_batch_shape(self) -> None:
        """Batch of N positions produces shape (N, 119, 8, 8)."""
        states = [BoardState.initial() for _ in range(4)]
        batch = encode_batch(states)
        assert batch.shape == (4, TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)

    def test_single_item_batch(self) -> None:
        """Batch of 1 should equal unsqueezed single encoding."""
        state = BoardState.initial()
        single = encode_board(state)
        batch = encode_batch([state])
        assert batch.shape == (1, TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert torch.allclose(batch[0], single)

    def test_batch_consistency(self) -> None:
        """Each element in the batch should match its individual encoding."""
        state1 = BoardState.initial()
        state2 = BoardState.empty()
        state2.pieces[(3, 3)] = (Color.WHITE, PieceType.QUEEN)

        batch = encode_batch([state1, state2])

        assert torch.allclose(batch[0], encode_board(state1))
        assert torch.allclose(batch[1], encode_board(state2))

    def test_empty_batch_raises(self) -> None:
        """Encoding an empty list should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            encode_batch([])

    def test_batch_dtype(self) -> None:
        """Batch tensor is float32."""
        states = [BoardState.initial()]
        batch = encode_batch(states)
        assert batch.dtype == torch.float32

    def test_batch_different_positions(self) -> None:
        """Batch encoding of different positions produces different tensors."""
        state1 = BoardState.initial()
        state2 = BoardState.empty()
        state2.pieces[(4, 4)] = (Color.BLACK, PieceType.KNIGHT)
        state2.side_to_move = Color.BLACK

        batch = encode_batch([state1, state2])
        # They should be different
        assert not torch.allclose(batch[0], batch[1])


# =============================================================================
# 10. Integration: Known Position Verification
# =============================================================================


class TestIntegration:
    """Hand-verify specific known positions for correctness."""

    def test_scholars_mate_position(self) -> None:
        """Encode the position just before Scholar's Mate (Qxf7#).

        FEN: r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4
        Black to move (but they're in checkmate).
        """
        fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        state = BoardState.from_fen_piece_placement(fen)
        tensor = encode_board(state)

        # Basic shape and dtype checks
        assert tensor.shape == (TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert tensor.dtype == torch.float32

        # Black to move -> color plane should be all 0s
        assert tensor[COLOR_PLANE].sum() == 0.0

        # Board is flipped because black is to move
        # The white queen on f7 (rank 6, file 5) is the opponent
        # After flip: rank 6 -> rank 1
        opp_queen = OPPONENT_PIECE_OFFSET + PieceType.QUEEN
        assert tensor[opp_queen, 1, 5] == 1.0

        # Black king on e8 (rank 7, file 4) is current player
        # After flip: rank 7 -> rank 0
        assert tensor[PieceType.KING, 0, 4] == 1.0

    def test_lone_kings_endgame(self) -> None:
        """K vs K position should have minimal non-zero planes."""
        state = BoardState.empty()
        state.pieces[(0, 4)] = (Color.WHITE, PieceType.KING)
        state.pieces[(7, 4)] = (Color.BLACK, PieceType.KING)
        state.side_to_move = Color.WHITE
        state.halfmove_clock = 50
        state.fullmove_number = 80

        tensor = encode_board(state)

        # Only 2 non-zero piece entries across all history planes
        piece_planes_sum = tensor[0:12].sum()
        assert piece_planes_sum == 2.0  # Two kings

        # Halfmove clock: 50/100 = 0.5
        expected_no_progress = 50.0 / 100.0
        assert torch.allclose(
            tensor[NO_PROGRESS_PLANE],
            torch.full((8, 8), expected_no_progress),
        )

        # Move count: 80/200 = 0.4
        expected_move_count = 80.0 / 200.0
        assert torch.allclose(
            tensor[MOVE_COUNT_PLANE],
            torch.full((8, 8), expected_move_count),
        )

    def test_encoding_is_deterministic(self) -> None:
        """Encoding the same position twice produces identical tensors."""
        state = BoardState.initial()
        t1 = encode_board(state)
        t2 = encode_board(state)
        assert torch.allclose(t1, t2)

    def test_mid_game_with_history(self) -> None:
        """Encode a position with history and verify all time steps are populated."""
        # Simulate a 4-move game with simplified states
        states = []
        for i in range(4):
            s = BoardState.empty()
            s.side_to_move = Color.WHITE if (i % 2 == 0) else Color.BLACK
            # Put a king somewhere unique for each state
            s.pieces[(i, 0)] = (Color.WHITE, PieceType.KING)
            s.pieces[(7 - i, 7)] = (Color.BLACK, PieceType.KING)
            states.append(s)

        # Latest state is the current position
        current = states[-1]
        current.history = list(reversed(states[:-1]))  # Most recent first

        tensor = encode_board(current)

        # 4 time steps should have pieces, 4 should be empty
        non_empty_steps = 0
        for t in range(HISTORY_STEPS):
            t_start = t * PLANES_PER_TIME_STEP
            piece_sum = tensor[t_start : t_start + 12].sum()
            if piece_sum > 0:
                non_empty_steps += 1

        assert non_empty_steps == 4

    def test_symmetric_encoding(self) -> None:
        """A symmetric position should produce a symmetric encoding (modulo color).

        Place a white queen on d4 and a black queen on d5. When white is to move,
        the white queen is "current player" and the black queen is "opponent".
        When black is to move (and the board is flipped), the roles reverse and
        the positions mirror.
        """
        # White to move
        state_w = BoardState.empty()
        state_w.pieces[(3, 3)] = (Color.WHITE, PieceType.QUEEN)  # d4
        state_w.pieces[(4, 3)] = (Color.BLACK, PieceType.QUEEN)  # d5
        state_w.side_to_move = Color.WHITE

        # Black to move (same piece positions)
        state_b = BoardState.empty()
        state_b.pieces[(3, 3)] = (Color.WHITE, PieceType.QUEEN)  # d4
        state_b.pieces[(4, 3)] = (Color.BLACK, PieceType.QUEEN)  # d5
        state_b.side_to_move = Color.BLACK

        tw = encode_board(state_w)
        tb = encode_board(state_b)

        # White to move: white queen (current) on (3,3), black queen (opponent) on (4,3)
        assert tw[PieceType.QUEEN, 3, 3] == 1.0
        assert tw[OPPONENT_PIECE_OFFSET + PieceType.QUEEN, 4, 3] == 1.0

        # Black to move: board flipped, so d4 (rank 3) -> rank 4, d5 (rank 4) -> rank 3
        # Black queen (current player, since black moves) was on (4,3), flips to (3,3)
        assert tb[PieceType.QUEEN, 3, 3] == 1.0
        # White queen (opponent) was on (3,3), flips to (4,3)
        assert tb[OPPONENT_PIECE_OFFSET + PieceType.QUEEN, 4, 3] == 1.0
