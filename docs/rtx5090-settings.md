# CurveAttention — larger-card notes

Card presets were removed in 2.5. They read as tuned and were not.

Use `cache_preset=native` or `fast` as an unvalidated starting point, then set `cache_max_mib` yourself. 2048–3072 is a guess for a 32 GB card, not a measured budget. `turbo_safe` is the short-schedule starting point.

Check the log for `CurveAttention cache: hits=`. Zero hits means the cache did no work. Do not stack with FirstBlockCache, TeaCache, SeaCache, or MotionCache.
