"""
Tests for Move Encoding/Decoding
=================================

These tests verify that the bidirectional mapping between chess moves and
policy vector indices is correct. We test at multiple levels:

1. Constants: verify the layout constants are internally consistent.
2. Queen moves: verify encoding/decoding for all 8 directions and 7 distances.
3. Knight moves: verify all 8 knight patterns encode/decode correctly.
4. Underpromotions: verify all 9 underpromotion types (3 pieces x 3 directions).
5. Queen promotions: verify they're encoded as regular queen-type moves.
6. Board flipping: verify rank mirroring for black's perspective.
7. Roundtrip: verify all 4672 indices roundtrip correctly.
8. Legal move mask: verify the boolean mask generation.
9. Edge cases: boundary conditions, invalid inputs.
10. Integration: encode real chess moves from known positions.
"""

import torch
import pytest

from neural.moves import (
    # Constants
    NUM_MOVE_TYPES,
    NUM_QUEEN_MOVE_TYPES,
    NUM_KNIGHT_MOVE_TYPES,
    NUM_UNDERPROMOTION_TYPES,
    POLICY_SIZE,
    BOARD_SIZE,
    QUEEN_DIRECTIONS,
    KNIGHT_DELTAS,
    UNDERPROMOTION_PIECES,
    UNDERPROMOTION_FILE_DELTAS,
    QUEEN_MOVE_OFFSET,
    KNIGHT_MOVE_OFFSET,
    UNDERPROMOTION_OFFSET,
    # Functions
    move_to_index,
    index_to_move,
    get_legal_move_mask,
    # Internal helpers (for white-box testing)
    _is_knight_move,
    _delta_to_queen_direction,
    _INDEX_TO_MOVE_TABLE,
)


# =============================================================================
# 1. Constant Consistency
# =============================================================================


class TestConstants:
    """Verify the layout constants are internally consistent."""

    def test_num_move_types_is_73(self) -> None:
        """73 move types per source square: 56 queen + 8 knight + 9 underpromotion."""
        assert NUM_MOVE_TYPES == 73

    def test_policy_size_is_4672(self) -> None:
        """8 * 8 * 73 = 4672 total policy outputs."""
        assert POLICY_SIZE == 8 * 8 * 73
        assert POLICY_SIZE == 4672

    def test_queen_move_count(self) -> None:
        """56 queen-type moves: 8 directions x 7 distances."""
        assert NUM_QUEEN_MOVE_TYPES == 56
        assert NUM_QUEEN_MOVE_TYPES == 8 * 7

    def test_knight_move_count(self) -> None:
        """8 knight move types."""
        assert NUM_KNIGHT_MOVE_TYPES == 8

    def test_underpromotion_count(self) -> None:
        """9 underpromotion types: 3 pieces x 3 directions."""
        assert NUM_UNDERPROMOTION_TYPES == 9
        assert NUM_UNDERPROMOTION_TYPES == 3 * 3

    def test_move_type_offsets(self) -> None:
        """Verify that the move type ranges don't overlap."""
        assert QUEEN_MOVE_OFFSET == 0
        assert KNIGHT_MOVE_OFFSET == 56
        assert UNDERPROMOTION_OFFSET == 64
        # Last move type index should be 72
        assert UNDERPROMOTION_OFFSET + NUM_UNDERPROMOTION_TYPES - 1 == 72

    def test_direction_count(self) -> None:
        """8 queen directions and 8 knight deltas."""
        assert len(QUEEN_DIRECTIONS) == 8
        assert len(KNIGHT_DELTAS) == 8

    def test_underpromotion_pieces(self) -> None:
        """Three underpromotion piece types: knight, bishop, rook."""
        assert UNDERPROMOTION_PIECES == ["knight", "bishop", "rook"]

    def test_underpromotion_file_deltas(self) -> None:
        """Three promotion directions: left (-1), straight (0), right (+1)."""
        assert UNDERPROMOTION_FILE_DELTAS == [-1, 0, +1]

    def test_index_table_size(self) -> None:
        """The precomputed table should have exactly 4672 entries."""
        assert len(_INDEX_TO_MOVE_TABLE) == POLICY_SIZE

    def test_policy_size_matches_config(self) -> None:
        """POLICY_SIZE must match NetworkConfig.policy_output_size."""
        from neural.config import NetworkConfig
        config = NetworkConfig()
        assert POLICY_SIZE == config.policy_output_size


# =============================================================================
# 2. Queen Move Encoding
# =============================================================================


class TestQueenMoves:
    """Verify encoding and decoding of queen-type moves."""

    def test_north_moves_from_e1(self) -> None:
        """Encode e1 moving north (up ranks) for each distance."""
        # e1 = rank 0, file 4
        for dist in range(1, 8):
            idx = move_to_index(0, 4, 0 + dist, 4)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (0, 4)
            assert (tr, tf) == (dist, 4)
            # No promotion for distances > 1 or not from rank 6
            if dist <= 6:  # not reaching rank 7 from rank 0 at dist <= 6
                assert promo is None

    def test_south_moves_from_e8(self) -> None:
        """Encode e8 (rank 7, file 4) moving south."""
        for dist in range(1, 8):
            idx = move_to_index(7, 4, 7 - dist, 4)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (7, 4)
            assert (tr, tf) == (7 - dist, 4)

    def test_east_moves_from_a4(self) -> None:
        """Encode a4 (rank 3, file 0) moving east (increasing file)."""
        for dist in range(1, 8):
            idx = move_to_index(3, 0, 3, dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (3, 0)
            assert (tr, tf) == (3, dist)
            assert promo is None

    def test_west_moves_from_h4(self) -> None:
        """Encode h4 (rank 3, file 7) moving west (decreasing file)."""
        for dist in range(1, 8):
            idx = move_to_index(3, 7, 3, 7 - dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (3, 7)
            assert (tr, tf) == (3, 7 - dist)
            assert promo is None

    def test_northeast_diagonal(self) -> None:
        """Encode a1 (rank 0, file 0) moving northeast (diagonal)."""
        for dist in range(1, 8):
            idx = move_to_index(0, 0, dist, dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (0, 0)
            assert (tr, tf) == (dist, dist)

    def test_southwest_diagonal(self) -> None:
        """Encode h8 (rank 7, file 7) moving southwest."""
        for dist in range(1, 8):
            idx = move_to_index(7, 7, 7 - dist, 7 - dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (7, 7)
            assert (tr, tf) == (7 - dist, 7 - dist)

    def test_southeast_diagonal(self) -> None:
        """Encode a8 (rank 7, file 0) moving southeast."""
        for dist in range(1, 7):
            idx = move_to_index(7, 0, 7 - dist, dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (7, 0)
            assert (tr, tf) == (7 - dist, dist)

    def test_northwest_diagonal(self) -> None:
        """Encode h1 (rank 0, file 7) moving northwest."""
        for dist in range(1, 8):
            idx = move_to_index(0, 7, dist, 7 - dist)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (0, 7)
            assert (tr, tf) == (dist, 7 - dist)

    def test_all_directions_from_d4(self) -> None:
        """From d4 (rank 3, file 3), encode one move in each direction."""
        expected_targets = [
            (4, 3),  # North
            (4, 4),  # Northeast
            (3, 4),  # East
            (2, 4),  # Southeast
            (2, 3),  # South
            (2, 2),  # Southwest
            (3, 2),  # West
            (4, 2),  # Northwest
        ]
        for dir_idx, (tr, tf) in enumerate(expected_targets):
            idx = move_to_index(3, 3, tr, tf)
            fr2, ff2, tr2, tf2, _ = index_to_move(idx)
            assert (fr2, ff2) == (3, 3), f"Direction {dir_idx}: wrong source"
            assert (tr2, tf2) == (tr, tf), f"Direction {dir_idx}: wrong target"

    def test_each_direction_has_7_distances(self) -> None:
        """From e4 (center-ish), verify 7 distinct indices per direction.

        We pick e4 (rank 3, file 4) which has room for several distances
        in most directions, but not all 7 in every direction. We just
        verify the indices are distinct.
        """
        from_rank, from_file = 3, 4
        all_indices: set = set()
        for dir_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
            for dist in range(1, 8):
                tr = from_rank + dr * dist
                tf = from_file + df * dist
                # Only encode if on-board
                if 0 <= tr < 8 and 0 <= tf < 8:
                    idx = move_to_index(from_rank, from_file, tr, tf)
                    assert idx not in all_indices, (
                        f"Duplicate index {idx} for direction {dir_idx}, distance {dist}"
                    )
                    all_indices.add(idx)

    def test_pawn_single_push_is_queen_move(self) -> None:
        """A pawn push (e.g., e2-e3) is encoded as a North queen move, distance 1."""
        idx = move_to_index(1, 4, 2, 4)  # e2-e3
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff, tr, tf) == (1, 4, 2, 4)
        assert promo is None

    def test_pawn_double_push_is_queen_move(self) -> None:
        """A pawn double push (e.g., e2-e4) is encoded as North queen move, distance 2."""
        idx = move_to_index(1, 4, 3, 4)  # e2-e4
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff, tr, tf) == (1, 4, 3, 4)
        assert promo is None

    def test_pawn_capture_is_queen_move(self) -> None:
        """A pawn diagonal capture is encoded as a queen-type diagonal move."""
        # White pawn on d4 captures on e5
        idx = move_to_index(3, 3, 4, 4)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff, tr, tf) == (3, 3, 4, 4)


# =============================================================================
# 3. Knight Move Encoding
# =============================================================================


class TestKnightMoves:
    """Verify encoding and decoding of knight moves."""

    def test_all_eight_knight_deltas(self) -> None:
        """From d4 (rank 3, file 3), verify all 8 knight moves encode correctly."""
        from_rank, from_file = 3, 3
        for delta_idx, (dr, df) in enumerate(KNIGHT_DELTAS):
            to_rank = from_rank + dr
            to_file = from_file + df
            assert 0 <= to_rank < 8 and 0 <= to_file < 8, (
                f"Knight delta {delta_idx} goes off board from (3,3)"
            )
            idx = move_to_index(from_rank, from_file, to_rank, to_file)
            fr, ff, tr, tf, promo = index_to_move(idx)
            assert (fr, ff) == (from_rank, from_file)
            assert (tr, tf) == (to_rank, to_file)
            assert promo is None

    def test_knight_from_g1_to_f3(self) -> None:
        """Classic opening move: Ng1-f3 (rank 0, file 6 -> rank 2, file 5)."""
        idx = move_to_index(0, 6, 2, 5)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (0, 6)
        assert (tr, tf) == (2, 5)
        assert promo is None

    def test_knight_from_b1_to_c3(self) -> None:
        """Classic opening move: Nb1-c3 (rank 0, file 1 -> rank 2, file 2)."""
        idx = move_to_index(0, 1, 2, 2)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (0, 1)
        assert (tr, tf) == (2, 2)
        assert promo is None

    def test_knight_moves_are_distinct_from_queen_moves(self) -> None:
        """Knight move indices should be different from queen move indices for the same square."""
        # From d4, a knight move to e6 is delta (+2, +1)
        knight_idx = move_to_index(3, 3, 5, 4)
        # From d4, a queen move to e5 (northeast, distance 1) is different
        queen_idx = move_to_index(3, 3, 4, 4)
        assert knight_idx != queen_idx

    def test_all_knight_indices_are_unique(self) -> None:
        """All 8 knight moves from a given square should produce unique indices."""
        from_rank, from_file = 3, 3
        indices = set()
        for dr, df in KNIGHT_DELTAS:
            tr = from_rank + dr
            tf = from_file + df
            if 0 <= tr < 8 and 0 <= tf < 8:
                idx = move_to_index(from_rank, from_file, tr, tf)
                assert idx not in indices
                indices.add(idx)

    def test_is_knight_move_helper(self) -> None:
        """Verify the _is_knight_move helper for all deltas."""
        # Valid knight moves
        for dr, df in KNIGHT_DELTAS:
            assert _is_knight_move(dr, df), f"({dr},{df}) should be a knight move"

        # Not knight moves
        assert not _is_knight_move(1, 0)   # North
        assert not _is_knight_move(1, 1)   # Northeast
        assert not _is_knight_move(0, 0)   # No move
        assert not _is_knight_move(3, 0)   # Too far
        assert not _is_knight_move(2, 2)   # Diagonal, not knight


# =============================================================================
# 4. Underpromotion Encoding
# =============================================================================


class TestUnderpromotions:
    """Verify encoding of underpromotions (knight, bishop, rook)."""

    def test_all_nine_underpromotion_types(self) -> None:
        """Verify all 9 underpromotion combinations from a7 (rank 6, file 0).

        From rank 6, a pawn can promote on rank 7 by:
        - Pushing straight to (7, 0)
        - Capturing right to (7, 1)
        (Can't capture left from file 0 -- off the board)
        We test straight and right here, and verify the encoding structure.
        """
        # Test all piece types with straight push from b7 (rank 6, file 1)
        # which allows all three directions
        from_rank, from_file = 6, 1
        seen_indices = set()

        for piece in UNDERPROMOTION_PIECES:
            for file_delta in UNDERPROMOTION_FILE_DELTAS:
                to_file = from_file + file_delta
                if 0 <= to_file < 8:
                    idx = move_to_index(
                        from_rank, from_file, 7, to_file,
                        promotion=piece,
                    )
                    assert idx not in seen_indices, (
                        f"Duplicate index for {piece}, delta={file_delta}"
                    )
                    seen_indices.add(idx)

                    fr, ff, tr, tf, promo = index_to_move(idx)
                    assert (fr, ff) == (from_rank, from_file)
                    assert (tr, tf) == (7, to_file)
                    assert promo == piece

    def test_knight_promotion_straight(self) -> None:
        """Promote to knight by pushing straight: e7-e8=N."""
        idx = move_to_index(6, 4, 7, 4, promotion="knight")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 4)
        assert promo == "knight"

    def test_bishop_promotion_capture_left(self) -> None:
        """Promote to bishop by capturing left: e7-d8=B."""
        idx = move_to_index(6, 4, 7, 3, promotion="bishop")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 3)
        assert promo == "bishop"

    def test_rook_promotion_capture_right(self) -> None:
        """Promote to rook by capturing right: e7-f8=R."""
        idx = move_to_index(6, 4, 7, 5, promotion="rook")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 5)
        assert promo == "rook"

    def test_underpromotion_distinct_from_queen_promotion(self) -> None:
        """A knight promotion must have a different index than a queen promotion of the same move."""
        # e7-e8: both queen and knight promotions
        queen_idx = move_to_index(6, 4, 7, 4, promotion="queen")
        knight_idx = move_to_index(6, 4, 7, 4, promotion="knight")
        assert queen_idx != knight_idx

    def test_all_underpromotion_pieces(self) -> None:
        """All three underpromotion pieces produce distinct indices for the same move."""
        indices = set()
        for piece in UNDERPROMOTION_PIECES:
            idx = move_to_index(6, 4, 7, 4, promotion=piece)
            assert idx not in indices
            indices.add(idx)
        assert len(indices) == 3


# =============================================================================
# 5. Queen Promotion Encoding
# =============================================================================


class TestQueenPromotion:
    """Verify that queen promotions are encoded as regular queen-type moves."""

    def test_queen_promotion_straight(self) -> None:
        """e7-e8=Q should be a North queen move, distance 1."""
        # With explicit "queen" promotion
        idx_explicit = move_to_index(6, 4, 7, 4, promotion="queen")
        # Without promotion specified (same move)
        idx_implicit = move_to_index(6, 4, 7, 4)
        # Both should produce the same index
        assert idx_explicit == idx_implicit

    def test_queen_promotion_decoded_as_queen(self) -> None:
        """Decoding a queen promotion move should indicate 'queen' promotion."""
        idx = move_to_index(6, 4, 7, 4, promotion="queen")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 4)
        # The index_to_move should detect this is a queen promotion
        # (rank 6 -> rank 7, distance 1)
        assert promo == "queen"

    def test_queen_promotion_capture_left(self) -> None:
        """e7-d8=Q (capture left) is a Northwest queen move."""
        idx = move_to_index(6, 4, 7, 3, promotion="queen")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 3)
        assert promo == "queen"

    def test_queen_promotion_capture_right(self) -> None:
        """e7-f8=Q (capture right) is a Northeast queen move."""
        idx = move_to_index(6, 4, 7, 5, promotion="queen")
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (6, 4)
        assert (tr, tf) == (7, 5)
        assert promo == "queen"


# =============================================================================
# 6. Board Flipping for Black
# =============================================================================


class TestBoardFlipping:
    """Verify that rank flipping works correctly for black's perspective."""

    def test_flip_simple_pawn_push(self) -> None:
        """Black's pawn on e7 (rank 6) pushing to e6 (rank 5).

        In real coordinates: from (6,4) to (5,4).
        After flipping: from (1,4) to (2,4) -- same as a white pawn push.
        """
        idx = move_to_index(6, 4, 5, 4, flip_for_black=True)
        # The index should match the white equivalent
        idx_white = move_to_index(1, 4, 2, 4)
        assert idx == idx_white

    def test_flip_roundtrip(self) -> None:
        """Encoding with flip and decoding with flip should recover the original move."""
        from_rank, from_file, to_rank, to_file = 6, 4, 5, 4
        idx = move_to_index(from_rank, from_file, to_rank, to_file, flip_for_black=True)
        fr, ff, tr, tf, promo = index_to_move(idx, flip_for_black=True)
        assert (fr, ff) == (from_rank, from_file)
        assert (tr, tf) == (to_rank, to_file)

    def test_flip_knight_move(self) -> None:
        """Black knight move with flipping should roundtrip correctly."""
        # Ng8-f6: from (7, 6) to (5, 5) in real coords
        idx = move_to_index(7, 6, 5, 5, flip_for_black=True)
        fr, ff, tr, tf, promo = index_to_move(idx, flip_for_black=True)
        assert (fr, ff) == (7, 6)
        assert (tr, tf) == (5, 5)
        assert promo is None

    def test_flip_underpromotion(self) -> None:
        """Black pawn promoting on rank 0 (real) maps to rank 7 (flipped).

        Black pawn on b2 (rank 1) promotes to knight on b1 (rank 0).
        After flipping: from rank 6 to rank 7 -- standard underpromotion.
        """
        idx = move_to_index(1, 1, 0, 1, promotion="knight", flip_for_black=True)
        fr, ff, tr, tf, promo = index_to_move(idx, flip_for_black=True)
        assert (fr, ff) == (1, 1)
        assert (tr, tf) == (0, 1)
        assert promo == "knight"

    def test_flip_queen_promotion(self) -> None:
        """Black pawn queen-promoting on rank 0 should also roundtrip."""
        idx = move_to_index(1, 4, 0, 4, promotion="queen", flip_for_black=True)
        fr, ff, tr, tf, promo = index_to_move(idx, flip_for_black=True)
        assert (fr, ff) == (1, 4)
        assert (tr, tf) == (0, 4)
        assert promo == "queen"

    def test_flip_maps_black_pawn_push_to_white_equivalent(self) -> None:
        """After flipping, a black pawn push has the same index as the white equivalent.

        Black e7-e5 (rank 6 file 4 to rank 4 file 4) flipped becomes
        (rank 1 file 4 to rank 3 file 4) = white e2-e4.
        """
        black_idx = move_to_index(6, 4, 4, 4, flip_for_black=True)
        white_idx = move_to_index(1, 4, 3, 4, flip_for_black=False)
        assert black_idx == white_idx


# =============================================================================
# 7. Full Roundtrip for All 4672 Indices
# =============================================================================


class TestRoundtrip:
    """Verify that every policy index roundtrips correctly through decode -> encode."""

    def test_all_4672_indices_roundtrip(self) -> None:
        """Decode each index and re-encode it; the result should be the same index.

        This is the definitive test: it proves the encoding is a bijection on
        the set of valid (on-board) moves, and that every index can be decoded
        and re-encoded without loss.
        """
        failed_indices = []
        for idx in range(POLICY_SIZE):
            fr, ff, tr, tf, promo = index_to_move(idx)
            # Some indices correspond to off-board moves (e.g., a queen move
            # of distance 7 from a corner). These can still be decoded, but
            # they can't be re-encoded because the to-square is off the board.
            if not (0 <= tr < 8 and 0 <= tf < 8):
                continue  # Skip off-board moves

            try:
                re_idx = move_to_index(fr, ff, tr, tf, promotion=promo)
                if re_idx != idx:
                    failed_indices.append(
                        (idx, re_idx, fr, ff, tr, tf, promo)
                    )
            except ValueError as e:
                failed_indices.append(
                    (idx, f"ERROR: {e}", fr, ff, tr, tf, promo)
                )

        assert len(failed_indices) == 0, (
            f"{len(failed_indices)} indices failed roundtrip. "
            f"First 5: {failed_indices[:5]}"
        )

    def test_on_board_indices_are_reachable(self) -> None:
        """Every on-board move in the lookup table should produce a valid index
        when re-encoded."""
        on_board_count = 0
        for idx in range(POLICY_SIZE):
            fr, ff, tr, tf, promo = index_to_move(idx)
            if 0 <= tr < 8 and 0 <= tf < 8:
                on_board_count += 1
                re_idx = move_to_index(fr, ff, tr, tf, promotion=promo)
                assert re_idx == idx

        # Sanity check: there should be a significant number of on-board moves
        # (certainly more than 1000 out of 4672)
        assert on_board_count > 1000, (
            f"Only {on_board_count} on-board indices found, expected >1000"
        )

    def test_roundtrip_with_flip(self) -> None:
        """Roundtrip with flip_for_black=True for a selection of moves."""
        test_moves = [
            (6, 4, 5, 4, None),         # Black pawn push
            (7, 6, 5, 5, None),         # Black knight
            (1, 3, 0, 3, "knight"),     # Black underpromotion
            (1, 4, 0, 4, "queen"),      # Black queen promotion
            (7, 0, 0, 0, None),         # Rook-like move full board
        ]
        for from_rank, from_file, to_rank, to_file, promo in test_moves:
            idx = move_to_index(
                from_rank, from_file, to_rank, to_file,
                promotion=promo, flip_for_black=True,
            )
            fr, ff, tr, tf, pr = index_to_move(idx, flip_for_black=True)
            assert (fr, ff) == (from_rank, from_file), (
                f"Failed for move ({from_rank},{from_file})->({to_rank},{to_file}) promo={promo}"
            )
            assert (tr, tf) == (to_rank, to_file)
            assert pr == promo


# =============================================================================
# 8. Legal Move Mask
# =============================================================================


class TestLegalMoveMask:
    """Verify the legal move mask generation."""

    def test_empty_move_list(self) -> None:
        """An empty move list should produce an all-False mask."""
        mask = get_legal_move_mask([])
        assert mask.shape == (POLICY_SIZE,)
        assert mask.dtype == torch.bool
        assert mask.sum() == 0

    def test_single_move(self) -> None:
        """A single move should set exactly one bit in the mask."""
        moves = [(1, 4, 3, 4, None)]  # e2-e4
        mask = get_legal_move_mask(moves)
        assert mask.sum() == 1
        idx = move_to_index(1, 4, 3, 4)
        assert mask[idx] is True or mask[idx].item() is True

    def test_multiple_moves(self) -> None:
        """Multiple moves should set the corresponding bits."""
        moves = [
            (1, 4, 2, 4, None),  # e2-e3
            (1, 4, 3, 4, None),  # e2-e4
            (0, 6, 2, 5, None),  # Ng1-f3
        ]
        mask = get_legal_move_mask(moves)
        assert mask.sum() == 3

        for from_r, from_f, to_r, to_f, promo in moves:
            idx = move_to_index(from_r, from_f, to_r, to_f, promotion=promo)
            assert mask[idx].item() is True

    def test_mask_shape_and_dtype(self) -> None:
        """Mask should be a boolean tensor of shape (4672,)."""
        mask = get_legal_move_mask([(0, 0, 1, 0, None)])
        assert mask.shape == (POLICY_SIZE,)
        assert mask.dtype == torch.bool

    def test_mask_with_flip(self) -> None:
        """Legal move mask with flip_for_black should encode moves correctly."""
        # Black pawn e7-e5: real coords (6,4)->(4,4)
        moves = [(6, 4, 4, 4, None)]
        mask = get_legal_move_mask(moves, flip_for_black=True)
        assert mask.sum() == 1

        # The index should match the flipped encoding
        idx = move_to_index(6, 4, 4, 4, flip_for_black=True)
        assert mask[idx].item() is True

    def test_mask_with_promotions(self) -> None:
        """Promotion moves should appear in the mask."""
        moves = [
            (6, 4, 7, 4, "queen"),   # e7-e8=Q
            (6, 4, 7, 4, "knight"),  # e7-e8=N
            (6, 4, 7, 4, "bishop"),  # e7-e8=B
            (6, 4, 7, 4, "rook"),    # e7-e8=R
        ]
        mask = get_legal_move_mask(moves)
        assert mask.sum() == 4

    def test_starting_position_white(self) -> None:
        """White's starting moves: 20 legal moves (16 pawn + 4 knight).

        Pawn moves: each of 8 pawns can push 1 or 2 squares = 16
        Knight moves: Nb1-a3, Nb1-c3, Ng1-f3, Ng1-h3 = 4
        Total: 20
        """
        legal_moves = []
        # Pawn single pushes
        for f in range(8):
            legal_moves.append((1, f, 2, f, None))
        # Pawn double pushes
        for f in range(8):
            legal_moves.append((1, f, 3, f, None))
        # Knight moves
        legal_moves.append((0, 1, 2, 0, None))  # Nb1-a3
        legal_moves.append((0, 1, 2, 2, None))  # Nb1-c3
        legal_moves.append((0, 6, 2, 5, None))  # Ng1-f3
        legal_moves.append((0, 6, 2, 7, None))  # Ng1-h3

        mask = get_legal_move_mask(legal_moves)
        assert mask.sum() == 20


# =============================================================================
# 9. Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test boundary conditions and error handling."""

    def test_index_out_of_range_negative(self) -> None:
        """Negative index should raise ValueError."""
        with pytest.raises(ValueError, match="Policy index"):
            index_to_move(-1)

    def test_index_out_of_range_too_high(self) -> None:
        """Index >= 4672 should raise ValueError."""
        with pytest.raises(ValueError, match="Policy index"):
            index_to_move(4672)

    def test_index_boundary_min(self) -> None:
        """Index 0 should decode successfully."""
        result = index_to_move(0)
        assert len(result) == 5

    def test_index_boundary_max(self) -> None:
        """Index 4671 should decode successfully."""
        result = index_to_move(4671)
        assert len(result) == 5

    def test_invalid_promotion_piece(self) -> None:
        """Invalid promotion string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid promotion"):
            move_to_index(6, 4, 7, 4, promotion="pawn")

    def test_zero_distance_move(self) -> None:
        """A move to the same square should raise ValueError."""
        with pytest.raises(ValueError):
            move_to_index(3, 3, 3, 3)

    def test_invalid_queen_direction(self) -> None:
        """A delta that's not along a queen direction or knight pattern should fail."""
        # (3, 3) to (5, 4) is delta (+2, +1) which IS a knight move -- OK
        # (3, 3) to (6, 4) is delta (+3, +1) which is neither queen nor knight
        with pytest.raises(ValueError):
            move_to_index(3, 3, 6, 4)

    def test_corner_square_a1(self) -> None:
        """Moves from a1 (rank 0, file 0) should encode the limited set of on-board moves."""
        # North: up to 7 squares
        for dist in range(1, 8):
            idx = move_to_index(0, 0, dist, 0)
            fr, ff, tr, tf, _ = index_to_move(idx)
            assert (fr, ff, tr, tf) == (0, 0, dist, 0)

        # East: up to 7 squares
        for dist in range(1, 8):
            idx = move_to_index(0, 0, 0, dist)
            fr, ff, tr, tf, _ = index_to_move(idx)
            assert (fr, ff, tr, tf) == (0, 0, 0, dist)

        # Northeast diagonal: up to 7 squares
        for dist in range(1, 8):
            idx = move_to_index(0, 0, dist, dist)
            fr, ff, tr, tf, _ = index_to_move(idx)
            assert (fr, ff, tr, tf) == (0, 0, dist, dist)

    def test_corner_square_h8(self) -> None:
        """Moves from h8 (rank 7, file 7) should encode correctly."""
        # South: up to 7 squares
        for dist in range(1, 8):
            idx = move_to_index(7, 7, 7 - dist, 7)
            fr, ff, tr, tf, _ = index_to_move(idx)
            assert (fr, ff, tr, tf) == (7, 7, 7 - dist, 7)


# =============================================================================
# 10. Helper Functions
# =============================================================================


class TestHelpers:
    """Test internal helper functions."""

    def test_delta_to_queen_direction_all_eight(self) -> None:
        """Verify that each direction maps correctly."""
        expected = {
            (1, 0): 0,    # North
            (1, 1): 1,    # Northeast
            (0, 1): 2,    # East
            (-1, 1): 3,   # Southeast
            (-1, 0): 4,   # South
            (-1, -1): 5,  # Southwest
            (0, -1): 6,   # West
            (1, -1): 7,   # Northwest
        }
        for (dr, df), expected_dir in expected.items():
            # Test with various distances
            for dist in range(1, 4):
                result = _delta_to_queen_direction(dr * dist, df * dist)
                assert result == expected_dir, (
                    f"Delta ({dr*dist},{df*dist}) should map to direction {expected_dir}, "
                    f"got {result}"
                )

    def test_delta_to_queen_direction_zero_raises(self) -> None:
        """Zero delta should raise ValueError."""
        with pytest.raises(ValueError, match="Zero delta"):
            _delta_to_queen_direction(0, 0)

    def test_is_knight_move_positive_cases(self) -> None:
        """All 8 knight deltas should return True."""
        for dr, df in KNIGHT_DELTAS:
            assert _is_knight_move(dr, df)

    def test_is_knight_move_negative_cases(self) -> None:
        """Non-knight deltas should return False."""
        non_knight = [
            (0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 1), (0, 2), (2, 0),
        ]
        for dr, df in non_knight:
            assert not _is_knight_move(dr, df), f"({dr},{df}) is not a knight move"


# =============================================================================
# 11. Integration: Real Chess Move Sequences
# =============================================================================


class TestIntegration:
    """Encode/decode moves from real chess games to verify correctness."""

    def test_italian_game_opening(self) -> None:
        """Verify encoding of the Italian Game opening moves.

        1. e4 e5 2. Nf3 Nc6 3. Bc4
        """
        # 1. e4: white pawn e2-e4
        idx = move_to_index(1, 4, 3, 4)
        fr, ff, tr, tf, _ = index_to_move(idx)
        assert (fr, ff, tr, tf) == (1, 4, 3, 4)

        # 1... e5: black pawn e7-e5 (flip for black)
        idx = move_to_index(6, 4, 4, 4, flip_for_black=True)
        fr, ff, tr, tf, _ = index_to_move(idx, flip_for_black=True)
        assert (fr, ff, tr, tf) == (6, 4, 4, 4)

        # 2. Nf3: white knight g1-f3
        idx = move_to_index(0, 6, 2, 5)
        fr, ff, tr, tf, _ = index_to_move(idx)
        assert (fr, ff, tr, tf) == (0, 6, 2, 5)

        # 2... Nc6: black knight b8-c6 (flip for black)
        idx = move_to_index(7, 1, 5, 2, flip_for_black=True)
        fr, ff, tr, tf, _ = index_to_move(idx, flip_for_black=True)
        assert (fr, ff, tr, tf) == (7, 1, 5, 2)

        # 3. Bc4: white bishop f1-c4
        idx = move_to_index(0, 5, 3, 2)
        fr, ff, tr, tf, _ = index_to_move(idx)
        assert (fr, ff, tr, tf) == (0, 5, 3, 2)

    def test_castling_kingside_white(self) -> None:
        """White kingside castling: Ke1-g1 (encoded as king move, distance 2 east).

        Note: castling is encoded as the king's move (e1 to g1 for kingside,
        e1 to c1 for queenside). The rook's movement is implicit.
        """
        # e1 to g1: rank 0, file 4 -> rank 0, file 6 = East, distance 2
        idx = move_to_index(0, 4, 0, 6)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (0, 4)
        assert (tr, tf) == (0, 6)
        assert promo is None

    def test_castling_queenside_white(self) -> None:
        """White queenside castling: Ke1-c1."""
        # e1 to c1: rank 0, file 4 -> rank 0, file 2 = West, distance 2
        idx = move_to_index(0, 4, 0, 2)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (0, 4)
        assert (tr, tf) == (0, 2)
        assert promo is None

    def test_castling_black_kingside_with_flip(self) -> None:
        """Black kingside castling: Ke8-g8, encoded with flip.

        Real coords: (7, 4) -> (7, 6)
        Flipped: (0, 4) -> (0, 6) = same as white kingside castling
        """
        idx = move_to_index(7, 4, 7, 6, flip_for_black=True)
        white_idx = move_to_index(0, 4, 0, 6)
        assert idx == white_idx

    def test_en_passant_white(self) -> None:
        """White en passant: pawn on e5 captures d6.

        Encoded as a normal diagonal move (Northeast or Northwest).
        """
        # e5 to d6: rank 4, file 4 -> rank 5, file 3 = Northwest, distance 1
        idx = move_to_index(4, 4, 5, 3)
        fr, ff, tr, tf, promo = index_to_move(idx)
        assert (fr, ff) == (4, 4)
        assert (tr, tf) == (5, 3)
        # En passant has no promotion
        assert promo is None

    def test_all_four_promotion_types(self) -> None:
        """From e7 to e8, all four promotion types should produce distinct indices."""
        indices = set()
        for promo in ["queen", "knight", "bishop", "rook"]:
            idx = move_to_index(6, 4, 7, 4, promotion=promo)
            assert idx not in indices
            indices.add(idx)
        assert len(indices) == 4

    def test_long_rook_move(self) -> None:
        """Rook on a1 moving to a8 (full board traverse)."""
        idx = move_to_index(0, 0, 7, 0)
        fr, ff, tr, tf, _ = index_to_move(idx)
        assert (fr, ff, tr, tf) == (0, 0, 7, 0)

    def test_long_bishop_move(self) -> None:
        """Bishop on a1 moving to h8 (full diagonal)."""
        idx = move_to_index(0, 0, 7, 7)
        fr, ff, tr, tf, _ = index_to_move(idx)
        assert (fr, ff, tr, tf) == (0, 0, 7, 7)

    def test_no_index_collision_all_on_board_moves(self) -> None:
        """Every distinct on-board move should map to a distinct index.

        This tests the injective property: different moves -> different indices.
        """
        seen: dict = {}  # index -> (from_rank, from_file, to_rank, to_file, promo)
        collisions = []

        for from_rank in range(8):
            for from_file in range(8):
                # Queen-type moves
                for dir_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
                    for dist in range(1, 8):
                        tr = from_rank + dr * dist
                        tf = from_file + df * dist
                        if 0 <= tr < 8 and 0 <= tf < 8:
                            promo = None
                            if tr == 7 and from_rank == 6 and dist == 1 and abs(df) <= 1:
                                promo = "queen"
                            idx = move_to_index(from_rank, from_file, tr, tf, promotion=promo)
                            move = (from_rank, from_file, tr, tf, promo)
                            if idx in seen and seen[idx] != move:
                                collisions.append((move, seen[idx], idx))
                            seen[idx] = move

                # Knight moves
                for dr, df in KNIGHT_DELTAS:
                    tr = from_rank + dr
                    tf = from_file + df
                    if 0 <= tr < 8 and 0 <= tf < 8:
                        idx = move_to_index(from_rank, from_file, tr, tf)
                        move = (from_rank, from_file, tr, tf, None)
                        if idx in seen and seen[idx] != move:
                            collisions.append((move, seen[idx], idx))
                        seen[idx] = move

                # Underpromotions (only from rank 6)
                if from_rank == 6:
                    for piece in UNDERPROMOTION_PIECES:
                        for file_delta in UNDERPROMOTION_FILE_DELTAS:
                            tf = from_file + file_delta
                            if 0 <= tf < 8:
                                idx = move_to_index(
                                    from_rank, from_file, 7, tf, promotion=piece
                                )
                                move = (from_rank, from_file, 7, tf, piece)
                                if idx in seen and seen[idx] != move:
                                    collisions.append((move, seen[idx], idx))
                                seen[idx] = move

        assert len(collisions) == 0, (
            f"{len(collisions)} index collisions found. First 5: {collisions[:5]}"
        )
