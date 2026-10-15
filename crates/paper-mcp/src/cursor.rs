// Copyright (c) 2026 Jason Brazeal (dev@jasonbrazeal.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//! Server-side cursor cache for paging ranked result lists.
//!
//! The first search call hands its complete ranked list to
//! [`CursorCache::start`], which keeps the list and returns the first page
//! plus an opaque cursor. Later calls pass that cursor to
//! [`CursorCache::next`] and read from the stored list, so pages stay
//! stable however the index changes in between. Lists expire after an idle
//! TTL, and the cache holds a bounded number of lists, evicting the least
//! recently used first.

use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasher as _, Hasher as _, RandomState};
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Default number of items per page.
pub const DEFAULT_PAGE_SIZE: NonZeroUsize = non_zero(20);
/// Default idle time after which a stored list expires.
pub const DEFAULT_CURSOR_TTL: Duration = Duration::from_mins(15);
/// Default maximum number of stored lists.
pub const DEFAULT_CURSOR_CAPACITY: NonZeroUsize = non_zero(64);

/// Convert a constant to [`NonZeroUsize`], mapping zero to one.
const fn non_zero(n: usize) -> NonZeroUsize {
    match NonZeroUsize::new(n) {
        Some(n) => n,
        None => NonZeroUsize::MIN,
    }
}

/// Separates the list key from the page offset inside a cursor string.
const CURSOR_SEPARATOR: char = '.';

/// Source of the current time, injectable so tests control expiry.
pub trait Clock: fmt::Debug + Send + Sync {
    /// Return the current instant.
    fn now(&self) -> Instant;
}

/// [`Clock`] backed by [`Instant::now`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Paging limits for a [`CursorCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorConfig {
    /// Items per page.
    pub page_size: NonZeroUsize,
    /// Idle time after which a stored list expires. Every page read resets it.
    pub ttl: Duration,
    /// Maximum number of stored lists.
    pub capacity: NonZeroUsize,
}

impl Default for CursorConfig {
    fn default() -> Self {
        Self {
            page_size: DEFAULT_PAGE_SIZE,
            ttl: DEFAULT_CURSOR_TTL,
            capacity: DEFAULT_CURSOR_CAPACITY,
        }
    }
}

/// One page of a ranked result list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page<T> {
    /// The page's items, in rank order.
    pub items: Vec<T>,
    /// Cursor for the following page, `None` on the last page.
    pub next_cursor: Option<String>,
}

/// Why a cursor could not be resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CursorError {
    /// The cursor's list sat idle past the TTL and was dropped.
    #[error("cursor has expired; rerun the search to page through fresh results")]
    Expired,
    /// The cursor was never issued, is malformed, or its list was evicted.
    #[error("cursor is unknown; rerun the search to page through fresh results")]
    Unknown,
}

/// A stored ranked list.
#[derive(Debug)]
struct Entry<T> {
    results: Vec<T>,
    last_used: Instant,
    /// Recency stamp from [`CursorCache::tick`]; the smallest is evicted first.
    recency: u64,
}

/// Bounded, expiring store of ranked result lists, paged by opaque cursors.
///
/// Reading a page with a cursor is idempotent: the same cursor returns the
/// same page until its list expires or is evicted.
#[derive(Debug)]
pub struct CursorCache<T> {
    config: CursorConfig,
    clock: Box<dyn Clock>,
    entries: HashMap<String, Entry<T>>,
    /// Keyed hasher whose output makes list keys unguessable.
    key_state: RandomState,
    /// Monotonic counter making every list key unique.
    sequence: u64,
    /// Monotonic counter ordering entries by last use.
    tick: u64,
}

impl<T: Clone> CursorCache<T> {
    /// Create an empty cache that reads time from the system clock.
    #[must_use]
    pub fn new(config: CursorConfig) -> Self {
        Self::with_clock(config, Box::new(SystemClock))
    }

    /// Create an empty cache that reads time from `clock`.
    #[must_use]
    pub fn with_clock(config: CursorConfig, clock: Box<dyn Clock>) -> Self {
        Self {
            config,
            clock,
            entries: HashMap::new(),
            key_state: RandomState::new(),
            sequence: 0,
            tick: 0,
        }
    }

    /// Store `results` and return its first page.
    ///
    /// A list that fits in one page is not stored and its page carries no
    /// cursor. Storing a list when the cache is full evicts the least
    /// recently used one.
    pub fn start(&mut self, results: Vec<T>) -> Page<T> {
        let page_size = self.config.page_size.get();
        if results.len() <= page_size {
            return Page {
                items: results,
                next_cursor: None,
            };
        }
        let now = self.clock.now();
        while self.entries.len() >= self.config.capacity.get() {
            self.evict_least_recent();
        }
        let key = self.new_key();
        let items = results[..page_size].to_vec();
        let next_cursor = Some(cursor_string(&key, page_size));
        let recency = self.next_tick();
        self.entries.insert(
            key,
            Entry {
                results,
                last_used: now,
                recency,
            },
        );
        Page { items, next_cursor }
    }

    /// Return the page that `cursor` points at.
    ///
    /// Reading a page resets its list's idle TTL and marks it most recently
    /// used.
    ///
    /// # Errors
    ///
    /// [`CursorError::Expired`] when the list sat idle past the TTL, and
    /// [`CursorError::Unknown`] when the cursor was never issued, is
    /// malformed, or its list was evicted.
    pub fn next(&mut self, cursor: &str) -> Result<Page<T>, CursorError> {
        let (key, offset) = cursor
            .rsplit_once(CURSOR_SEPARATOR)
            .ok_or(CursorError::Unknown)?;
        let offset: usize = offset.parse().map_err(|_| CursorError::Unknown)?;
        let now = self.clock.now();
        let ttl = self.config.ttl;
        let page_size = self.config.page_size.get();
        let entry = self.entries.get(key).ok_or(CursorError::Unknown)?;
        if now.saturating_duration_since(entry.last_used) > ttl {
            self.entries.remove(key);
            return Err(CursorError::Expired);
        }
        // Issued cursors only ever point at a later page boundary inside the list.
        if offset == 0 || !offset.is_multiple_of(page_size) || offset >= entry.results.len() {
            return Err(CursorError::Unknown);
        }
        let recency = self.next_tick();
        let entry = self.entries.get_mut(key).ok_or(CursorError::Unknown)?;
        entry.last_used = now;
        entry.recency = recency;
        let end = offset.saturating_add(page_size).min(entry.results.len());
        let items = entry.results[offset..end].to_vec();
        let next_cursor = (end < entry.results.len()).then(|| cursor_string(key, end));
        Ok(Page { items, next_cursor })
    }

    fn evict_least_recent(&mut self) {
        let oldest = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.recency)
            .map(|(key, _)| key.clone());
        if let Some(key) = oldest {
            self.entries.remove(&key);
        }
    }

    /// Return a key no other list in this cache has used.
    ///
    /// The sequence number guarantees uniqueness; the keyed hash of it,
    /// whose key is random per process, makes keys hard to guess.
    fn new_key(&mut self) -> String {
        self.sequence = self.sequence.wrapping_add(1);
        let mut hasher = self.key_state.build_hasher();
        hasher.write_u64(self.sequence);
        format!("{:016x}{:x}", hasher.finish(), self.sequence)
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }
}

/// Render the cursor for the page of list `key` that starts at `offset`.
fn cursor_string(key: &str, offset: usize) -> String {
    format!("{key}{CURSOR_SEPARATOR}{offset}")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    /// Clock that only moves when a test advances it.
    #[derive(Clone, Debug)]
    struct ManualClock {
        base: Instant,
        elapsed_ms: Arc<AtomicU64>,
    }

    impl ManualClock {
        fn new() -> Self {
            Self {
                base: Instant::now(),
                elapsed_ms: Arc::new(AtomicU64::new(0)),
            }
        }

        fn advance(&self, by: Duration) {
            let ms = u64::try_from(by.as_millis()).unwrap_or(u64::MAX);
            self.elapsed_ms.fetch_add(ms, Ordering::SeqCst);
        }
    }

    impl Clock for ManualClock {
        fn now(&self) -> Instant {
            self.base + Duration::from_millis(self.elapsed_ms.load(Ordering::SeqCst))
        }
    }

    fn size(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).unwrap_or(NonZeroUsize::MIN)
    }

    fn cache(page_size: usize, capacity: usize) -> (CursorCache<u32>, ManualClock) {
        let clock = ManualClock::new();
        let config = CursorConfig {
            page_size: size(page_size),
            ttl: Duration::from_secs(60),
            capacity: size(capacity),
        };
        (
            CursorCache::with_clock(config, Box::new(clock.clone())),
            clock,
        )
    }

    fn items(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    /// Page `first` to its end, returning every page's items.
    fn drain(cache: &mut CursorCache<u32>, first: Page<u32>) -> Result<Vec<Vec<u32>>, CursorError> {
        let mut pages = vec![first.items];
        let mut cursor = first.next_cursor;
        while let Some(c) = cursor {
            let page = cache.next(&c)?;
            pages.push(page.items);
            cursor = page.next_cursor;
        }
        Ok(pages)
    }

    #[test]
    fn defaults_match_functional_specification() {
        let config = CursorConfig::default();
        assert_eq!(config.page_size.get(), 20);
        assert_eq!(config.ttl, Duration::from_mins(15));
        assert_eq!(config.capacity.get(), 64);
    }

    #[test]
    fn empty_results_give_one_empty_page_without_cursor() {
        let (mut cache, _) = cache(3, 4);
        let page = cache.start(Vec::new());
        assert!(page.items.is_empty());
        assert_eq!(page.next_cursor, None);
    }

    #[test]
    fn results_within_one_page_have_no_cursor() {
        let (mut cache, _) = cache(3, 4);
        assert_eq!(
            cache.start(items(2)),
            Page {
                items: vec![0, 1],
                next_cursor: None
            }
        );
        assert_eq!(
            cache.start(items(3)),
            Page {
                items: vec![0, 1, 2],
                next_cursor: None
            }
        );
    }

    #[test]
    fn exact_multiple_of_page_size_ends_without_trailing_empty_page() -> Result<(), CursorError> {
        let (mut cache, _) = cache(3, 4);
        let first = cache.start(items(6));
        assert_eq!(
            drain(&mut cache, first)?,
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
        Ok(())
    }

    #[test]
    fn paging_returns_every_item_once_in_rank_order() -> Result<(), CursorError> {
        let (mut cache, _) = cache(3, 4);
        let first = cache.start(items(8));
        assert_eq!(
            drain(&mut cache, first)?,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7]]
        );
        Ok(())
    }

    #[test]
    fn rereading_a_cursor_returns_the_same_page() -> Result<(), CursorError> {
        let (mut cache, _) = cache(2, 4);
        let first = cache.start(items(5));
        let cursor = first.next_cursor.ok_or(CursorError::Unknown)?;
        let once = cache.next(&cursor)?;
        let following = once.next_cursor.clone().ok_or(CursorError::Unknown)?;
        cache.next(&following)?;
        assert_eq!(cache.next(&cursor)?, once);
        Ok(())
    }

    #[test]
    fn pages_stay_stable_while_other_lists_are_stored() -> Result<(), CursorError> {
        let (mut cache, _) = cache(2, 4);
        let first = cache.start(vec![10, 11, 12, 13]);
        let cursor = first.next_cursor.ok_or(CursorError::Unknown)?;
        cache.start(vec![20, 21, 22, 23]);
        assert_eq!(cache.next(&cursor)?.items, vec![12, 13]);
        Ok(())
    }

    #[test]
    fn cursors_from_different_lists_differ() {
        let (mut cache, _) = cache(1, 4);
        let a = cache.start(items(2)).next_cursor;
        let b = cache.start(items(2)).next_cursor;
        assert!(a.is_some());
        assert_ne!(a, b);
    }

    #[test]
    fn idle_list_expires_after_ttl() {
        let (mut cache, clock) = cache(1, 4);
        let first = cache.start(items(3));
        clock.advance(Duration::from_secs(61));
        let cursor = first.next_cursor.unwrap_or_default();
        assert_eq!(cache.next(&cursor), Err(CursorError::Expired));
    }

    #[test]
    fn expired_list_reports_expired_after_another_search() {
        let (mut cache, clock) = cache(1, 4);
        let first = cache.start(items(3));
        clock.advance(Duration::from_secs(61));
        cache.start(items(3));
        let cursor = first.next_cursor.unwrap_or_default();
        assert_eq!(cache.next(&cursor), Err(CursorError::Expired));
    }

    #[test]
    fn reading_a_page_resets_the_idle_ttl() -> Result<(), CursorError> {
        let (mut cache, clock) = cache(1, 4);
        let first = cache.start(items(3));
        let cursor = first.next_cursor.ok_or(CursorError::Unknown)?;
        clock.advance(Duration::from_secs(40));
        let second = cache.next(&cursor)?;
        clock.advance(Duration::from_secs(40));
        let following = second.next_cursor.ok_or(CursorError::Unknown)?;
        assert_eq!(cache.next(&following)?.items, vec![2]);
        Ok(())
    }

    #[test]
    fn list_used_at_exactly_ttl_is_still_live() -> Result<(), CursorError> {
        let (mut cache, clock) = cache(1, 4);
        let first = cache.start(items(2));
        clock.advance(Duration::from_secs(60));
        let cursor = first.next_cursor.ok_or(CursorError::Unknown)?;
        assert_eq!(cache.next(&cursor)?.items, vec![1]);
        Ok(())
    }

    #[test]
    fn full_cache_evicts_least_recently_used_list() -> Result<(), CursorError> {
        let (mut cache, _) = cache(1, 2);
        let a = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        let b = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        // Touching `a` leaves `b` as the least recently used list.
        cache.next(&a)?;
        let c = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        assert_eq!(cache.next(&b), Err(CursorError::Unknown));
        assert_eq!(cache.next(&a)?.items, vec![1]);
        assert_eq!(cache.next(&c)?.items, vec![1]);
        Ok(())
    }

    #[test]
    fn eviction_without_reads_drops_the_oldest_list() -> Result<(), CursorError> {
        let (mut cache, _) = cache(1, 2);
        let a = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        let b = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        let c = cache
            .start(items(2))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        assert_eq!(cache.next(&a), Err(CursorError::Unknown));
        assert_eq!(cache.next(&b)?.items, vec![1]);
        assert_eq!(cache.next(&c)?.items, vec![1]);
        Ok(())
    }

    #[test]
    fn single_page_lists_do_not_take_capacity() -> Result<(), CursorError> {
        let (mut cache, _) = cache(2, 1);
        let a = cache
            .start(items(3))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        cache.start(items(2));
        cache.start(Vec::new());
        assert_eq!(cache.next(&a)?.items, vec![2]);
        Ok(())
    }

    #[test]
    fn unknown_cursor_is_rejected() {
        let (mut cache, _) = cache(1, 4);
        assert_eq!(cache.next("never-issued"), Err(CursorError::Unknown));
        assert_eq!(cache.next(""), Err(CursorError::Unknown));
    }

    #[test]
    fn tampered_offset_is_rejected() -> Result<(), CursorError> {
        let (mut cache, _) = cache(2, 4);
        let cursor = cache
            .start(items(5))
            .next_cursor
            .ok_or(CursorError::Unknown)?;
        let (key, _) = cursor
            .rsplit_once(CURSOR_SEPARATOR)
            .ok_or(CursorError::Unknown)?;
        for offset in ["0", "1", "6", "x", ""] {
            let forged = format!("{key}{CURSOR_SEPARATOR}{offset}");
            assert_eq!(cache.next(&forged), Err(CursorError::Unknown), "{forged}");
        }
        Ok(())
    }

    #[test]
    fn cursor_errors_tell_the_client_to_rerun_the_search() {
        for error in [CursorError::Expired, CursorError::Unknown] {
            assert!(error.to_string().contains("rerun the search"), "{error}");
        }
    }
}
