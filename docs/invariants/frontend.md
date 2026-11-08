# Frontend (Next.js)

- Next.js + TypeScript in its own folder (`web/`), managed with `pnpm`.
- Design for the real user: imagine who uses it and what they need to see
  first. The UI tells the story, not just renders data.
- Every number shown comes from the backend (see "Real data only" in core.md).
- Every request shows loading, error and empty states; the user can always recover.
- Skills: visual redesign → ui-ux-pro-max; review/polish → impeccable.
- A grid or flex track that holds user text is `minmax(0, 1fr)` (flex items `min-width: 0`); a bare `1fr` lets one long title widen the whole page on phones.
- Sticky chrome has one height token (e.g. `--banner-h`), and every sticky offset and `scroll-padding-top` derives from it.
- Behind a dev proxy (Next rewrites), a non-JSON error response means the backend is unreachable; say "start the backend", not "server error".
- Hover-only linking (highlight on hover) checks `pointerType === "mouse"` so taps don't leave sticky highlights.
