//! # Chess Engine
//!
//! A chess engine designed for AlphaZero-style self-play and training.
//! Prioritizes clarity and correctness over raw speed, though performance
//! is still important for generating training data.

pub mod attacks;
pub mod bitboard;
pub mod board;
pub mod fen;
pub mod game;
pub mod magic;
pub mod makemove;
pub mod movegen;
pub mod moves;
pub mod perft;
pub mod types;
pub mod zobrist;
