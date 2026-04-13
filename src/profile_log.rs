//! Non-blocking profile logging for IBD and consensus hot paths.
//!
//! Replaces `eprintln!` with channel-based logging so validation never blocks on I/O.
//! A dedicated background thread drains the channel and writes to stderr.
//! Note: `tracing` also uses stderr; without a global lock, lines can rarely interleave with
//! timestamps (`evict_ms=0` + `2026-03-29T...` on one physical line). Log analyzers normalize this;
//! prefer `analyze_ibd_profile.py` over naive `grep` for `[IBD_VALIDATION]` phase=end splits.

use std::io::Write;
use std::sync::mpsc;
use std::sync::OnceLock;

/// Capacity for the profile log channel. When full, new messages are dropped (non-blocking).
const CHANNEL_CAPACITY: usize = 65_536;

static LOGGER: OnceLock<mpsc::SyncSender<String>> = OnceLock::new();

#[allow(dead_code)]
pub fn sender() -> Option<&'static mpsc::SyncSender<String>> {
    Some(LOGGER.get_or_init(|| {
        let (tx, rx) = mpsc::sync_channel(CHANNEL_CAPACITY);
        std::thread::Builder::new()
            .name("profile-log".into())
            .spawn(move || {
                for msg in rx {
                    let _ = writeln!(std::io::stderr(), "{msg}");
                }
            })
            .expect("Failed to spawn profile log thread");
        tx
    }))
}

/// Log a profile message without blocking. Drops the message if the channel is full.
#[macro_export]
macro_rules! profile_log {
    ($($arg:tt)*) => {{
        #[cfg(feature = "profile")]
        {
            if let Some(tx) = $crate::profile_log::sender() {
                let _ = tx.try_send(format!($($arg)*));
            }
        }
    }};
}
