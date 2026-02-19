//! Directory-based replay buffer for self-play training data.
//!
//! Games are stored as individual MessagePack files in a directory. The buffer
//! maintains a configurable maximum capacity (number of game files) and evicts
//! the oldest files when the limit is reached.
//!
//! # File naming
//!
//! Game files are named `game_{timestamp_nanos}_{random}.msgpack` to ensure:
//! - Chronological ordering (oldest-first when sorted alphabetically)
//! - No collisions even when multiple workers write simultaneously
//!
//! # Usage
//!
//! ```no_run
//! use std::path::PathBuf;
//! use self_play::buffer::ReplayBuffer;
//!
//! let buffer = ReplayBuffer::new(PathBuf::from("/tmp/replay"), 1000).unwrap();
//! // buffer.add_game(&samples).unwrap();
//! // let batch = buffer.sample(256).unwrap();
//! ```

use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use rand::seq::SliceRandom;
use rand::Rng;

use crate::data::TrainingSample;
use crate::serialize::{read_game_file, write_game_file};

// =============================================================================
// ReplayBuffer
// =============================================================================

/// A directory-based replay buffer for self-play training data.
///
/// Games are stored as individual MessagePack files in a directory.
/// The buffer maintains a maximum capacity (number of games) and
/// evicts the oldest files when the limit is reached.
pub struct ReplayBuffer {
    /// Directory where game files are stored.
    dir: PathBuf,
    /// Maximum number of game files to keep.
    capacity: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer writing to the given directory.
    ///
    /// Creates the directory (and all parent directories) if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory to store game files in.
    /// * `capacity` - Maximum number of game files to keep. When exceeded,
    ///   the oldest files are removed.
    pub fn new(dir: PathBuf, capacity: usize) -> io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir, capacity })
    }

    /// Add a game's training samples to the buffer.
    ///
    /// Writes a new game file and evicts the oldest files if the buffer is
    /// over capacity.
    ///
    /// File naming: `game_{timestamp_nanos}_{random}.msgpack`
    ///
    /// Returns the path to the newly created file.
    pub fn add_game(&self, samples: &[TrainingSample]) -> io::Result<PathBuf> {
        let filename = generate_filename();
        let path = self.dir.join(&filename);
        write_game_file(&path, samples)?;
        self.evict()?;
        Ok(path)
    }

    /// List all game files in the buffer directory, sorted by name (oldest first).
    ///
    /// Only files with the `.msgpack` extension are included.
    pub fn list_games(&self) -> io::Result<Vec<PathBuf>> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&self.dir)?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|ext| ext.to_str()) == Some("msgpack") {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();
        // Sort by filename (which includes timestamp, so oldest first)
        files.sort();
        Ok(files)
    }

    /// Count the current number of games in the buffer.
    pub fn len(&self) -> io::Result<usize> {
        Ok(self.list_games()?.len())
    }

    /// Returns true if the buffer contains no game files.
    pub fn is_empty(&self) -> io::Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Sample N random training positions from random games in the buffer.
    ///
    /// Games are selected with replacement (the same game may be sampled
    /// multiple times). Within each selected game, a single random position
    /// is chosen.
    ///
    /// If the buffer is empty, returns an empty vector.
    /// If a game file cannot be read, it is skipped silently.
    pub fn sample(&self, count: usize) -> io::Result<Vec<TrainingSample>> {
        let games = self.list_games()?;
        if games.is_empty() {
            return Ok(Vec::new());
        }

        let mut rng = rand::thread_rng();
        let mut result = Vec::with_capacity(count);

        // Keep trying until we have enough samples or exhaust attempts
        let max_attempts = count * 3; // Allow some failures
        let mut attempts = 0;

        while result.len() < count && attempts < max_attempts {
            attempts += 1;

            // Pick a random game file
            let game_path = games.choose(&mut rng).unwrap();

            // Try to read the game file
            let samples = match read_game_file(game_path) {
                Ok(s) => s,
                Err(_) => continue, // Skip unreadable files
            };

            if samples.is_empty() {
                continue;
            }

            // Pick a random position from the game
            let idx = rng.gen_range(0..samples.len());
            result.push(samples[idx].clone());
        }

        Ok(result)
    }

    /// Return the directory path of this buffer.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Evict oldest game files to stay within capacity.
    ///
    /// Files are sorted by name (which encodes timestamp), and the oldest
    /// are removed first until the count is within capacity.
    ///
    /// Tolerates concurrent deletes: if another thread already removed a
    /// file, the `NotFound` error is silently ignored.
    fn evict(&self) -> io::Result<()> {
        let files = self.list_games()?;
        if files.len() <= self.capacity {
            return Ok(());
        }
        let to_remove = files.len() - self.capacity;
        for path in &files[..to_remove] {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Generate a unique filename for a game file.
///
/// Format: `game_{timestamp_nanos}_{random_u32}.msgpack`
fn generate_filename() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let random: u32 = rand::thread_rng().gen();
    format!("game_{}_{:08x}.msgpack", timestamp, random)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chess_engine::board::Board;
    use chess_engine::moves::Move;
    use chess_engine::types::Square;

    /// Helper: create a simple TrainingSample.
    fn make_sample(value: f32) -> TrainingSample {
        TrainingSample {
            board: Board::starting_position(),
            policy: vec![
                (Move::new(Square::E2, Square::E4), 0.6),
                (Move::new(Square::D2, Square::D4), 0.4),
            ],
            value,
        }
    }

    /// Helper: create a temporary directory for testing.
    fn temp_buffer_dir(test_name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join("alphazero_buffer_tests")
            .join(test_name);
        // Clean up any previous test run
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    // ========================================================================
    // 1. New buffer creates directory
    // ========================================================================

    #[test]
    fn new_buffer_creates_directory() {
        let dir = temp_buffer_dir("new_creates_dir");
        assert!(!dir.exists(), "Directory should not exist before creation");

        let _buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();
        assert!(dir.exists(), "Directory should exist after creation");
        assert!(dir.is_dir(), "Should be a directory");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 2. add_game creates a file
    // ========================================================================

    #[test]
    fn add_game_creates_file() {
        let dir = temp_buffer_dir("add_game_creates");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        let samples = vec![make_sample(1.0), make_sample(-1.0)];
        let path = buffer.add_game(&samples).unwrap();

        assert!(path.exists(), "Game file should exist");
        assert!(
            path.extension().unwrap() == "msgpack",
            "File should have .msgpack extension"
        );
        assert!(
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("game_"),
            "File should start with 'game_'"
        );

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 3. list_games returns files sorted oldest-first
    // ========================================================================

    #[test]
    fn list_games_sorted_oldest_first() {
        let dir = temp_buffer_dir("list_sorted");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        let samples = vec![make_sample(0.0)];

        // Add 3 games with slight delays to ensure different timestamps
        let path1 = buffer.add_game(&samples).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let path2 = buffer.add_game(&samples).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let path3 = buffer.add_game(&samples).unwrap();

        let listed = buffer.list_games().unwrap();
        assert_eq!(listed.len(), 3, "Should have 3 games");

        // Should be sorted oldest-first (by filename which includes timestamp)
        assert_eq!(listed[0], path1, "First should be oldest");
        assert_eq!(listed[1], path2, "Second should be middle");
        assert_eq!(listed[2], path3, "Third should be newest");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 4. Capacity eviction: oldest are removed
    // ========================================================================

    #[test]
    fn capacity_eviction_removes_oldest() {
        let dir = temp_buffer_dir("eviction");
        let buffer = ReplayBuffer::new(dir.clone(), 3).unwrap();

        let samples = vec![make_sample(0.0)];

        // Add 5 games with a capacity of 3
        for _ in 0..5 {
            buffer.add_game(&samples).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(2));
        }

        let count = buffer.len().unwrap();
        assert_eq!(
            count, 3,
            "Should have at most 3 games after eviction, got {}",
            count
        );

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 5. sample returns correct number of samples
    // ========================================================================

    #[test]
    fn sample_returns_correct_count() {
        let dir = temp_buffer_dir("sample_count");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        // Add a few games
        for v in &[1.0, -1.0, 0.0] {
            let samples = vec![make_sample(*v), make_sample(*v)];
            buffer.add_game(&samples).unwrap();
        }

        let sampled = buffer.sample(5).unwrap();
        assert_eq!(
            sampled.len(),
            5,
            "Should return 5 samples, got {}",
            sampled.len()
        );

        // Each sample should have valid data
        for sample in &sampled {
            assert!(!sample.policy.is_empty(), "Policy should not be empty");
            assert!(
                sample.value == 1.0 || sample.value == -1.0 || sample.value == 0.0,
                "Value should be in {{-1, 0, +1}}, got {}",
                sample.value
            );
        }

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 6. len() tracks game count
    // ========================================================================

    #[test]
    fn len_tracks_game_count() {
        let dir = temp_buffer_dir("len_tracks");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        assert_eq!(buffer.len().unwrap(), 0, "Should start empty");

        let samples = vec![make_sample(0.0)];
        buffer.add_game(&samples).unwrap();
        assert_eq!(buffer.len().unwrap(), 1, "Should have 1 after adding 1");

        buffer.add_game(&samples).unwrap();
        assert_eq!(buffer.len().unwrap(), 2, "Should have 2 after adding 2");

        buffer.add_game(&samples).unwrap();
        assert_eq!(buffer.len().unwrap(), 3, "Should have 3 after adding 3");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 7. Empty buffer returns empty sample
    // ========================================================================

    #[test]
    fn empty_buffer_returns_empty_sample() {
        let dir = temp_buffer_dir("empty_sample");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        let sampled = buffer.sample(10).unwrap();
        assert!(
            sampled.is_empty(),
            "Sampling from empty buffer should return empty vec"
        );

        assert!(buffer.is_empty().unwrap(), "Buffer should report as empty");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 8. Multiple adds and reads work correctly
    // ========================================================================

    #[test]
    fn multiple_adds_and_reads() {
        let dir = temp_buffer_dir("multi_add_read");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        // Add 10 games, each with different values
        for i in 0..10 {
            let value = if i % 3 == 0 {
                1.0
            } else if i % 3 == 1 {
                -1.0
            } else {
                0.0
            };
            let samples = vec![make_sample(value)];
            buffer.add_game(&samples).unwrap();
        }

        assert_eq!(buffer.len().unwrap(), 10, "Should have 10 games");

        // Read all games
        let games = buffer.list_games().unwrap();
        for game_path in &games {
            let samples = read_game_file(game_path).unwrap();
            assert_eq!(samples.len(), 1, "Each game has 1 sample");
        }

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 9. File naming format is correct
    // ========================================================================

    #[test]
    fn file_naming_format() {
        let filename = generate_filename();
        assert!(
            filename.starts_with("game_"),
            "Filename should start with 'game_'"
        );
        assert!(
            filename.ends_with(".msgpack"),
            "Filename should end with '.msgpack'"
        );

        // Should have format: game_{digits}_{hex}.msgpack
        let stem = filename.strip_prefix("game_").unwrap();
        let stem = stem.strip_suffix(".msgpack").unwrap();
        let parts: Vec<&str> = stem.rsplitn(2, '_').collect();
        assert_eq!(parts.len(), 2, "Should have timestamp_random format");

        // The random part should be 8 hex characters
        assert_eq!(
            parts[0].len(),
            8,
            "Random part should be 8 hex chars, got '{}'",
            parts[0]
        );
    }

    // ========================================================================
    // 10. Eviction preserves newest files
    // ========================================================================

    #[test]
    fn eviction_preserves_newest() {
        let dir = temp_buffer_dir("eviction_newest");
        let buffer = ReplayBuffer::new(dir.clone(), 2).unwrap();

        let samples = vec![make_sample(0.0)];

        // Add 3 games
        buffer.add_game(&samples).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let path2 = buffer.add_game(&samples).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let path3 = buffer.add_game(&samples).unwrap();

        let listed = buffer.list_games().unwrap();
        assert_eq!(listed.len(), 2, "Should have 2 games (capacity)");

        // The remaining files should be the 2 newest
        assert_eq!(listed[0], path2, "Should keep second game");
        assert_eq!(listed[1], path3, "Should keep third (newest) game");

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }

    // ========================================================================
    // 11. Non-msgpack files are ignored
    // ========================================================================

    #[test]
    fn non_msgpack_files_ignored() {
        let dir = temp_buffer_dir("non_msgpack");
        let buffer = ReplayBuffer::new(dir.clone(), 100).unwrap();

        // Create a non-msgpack file in the directory
        std::fs::write(dir.join("readme.txt"), "not a game file").unwrap();
        std::fs::write(dir.join("data.json"), "{}").unwrap();

        let samples = vec![make_sample(0.0)];
        buffer.add_game(&samples).unwrap();

        let listed = buffer.list_games().unwrap();
        assert_eq!(
            listed.len(),
            1,
            "Should only list .msgpack files, got {}",
            listed.len()
        );

        assert_eq!(buffer.len().unwrap(), 1);

        // Clean up
        std::fs::remove_dir_all(&dir).ok();
    }
}
