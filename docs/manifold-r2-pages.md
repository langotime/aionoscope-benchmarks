# Manifold Viewer Deployment: Cloudflare Pages + R2

This document is the durable source of truth for the hosted manifold viewer.
Future agents should read it before changing `results/manifolds.html`, uploading
manifold JSON data, or touching the Cloudflare setup.

## Architecture

The manifold deployment intentionally splits UI and data:

- Cloudflare Pages serves the checked-in static viewer from Git.
- Cloudflare R2 stores the generated manifold JSON payloads.
- Cloudflare Cache Rules make versioned R2 JSON cacheable at the edge.

Do not put the full `results/manifolds/` data directory into Git or Pages. It is
large generated output and is ignored by `.gitignore`.

## Cloudflare Resources

Cloudflare account:

- Account name: `Alexander.chemeris@gmail.com's Account`
- Account ID: `ac03aae0bfec2c5699adb479f5f4c5db`

Pages:

- Project: `aionoscope-benchmarks`
- Pages domain: `https://aionoscope-benchmarks.pages.dev`
- Git provider: enabled
- Production branch: `main`
- Production deployment source is Git, so checked-in files under `results/`
  are deployed by Cloudflare Pages after they land on `main`.

R2:

- Bucket: `aionoscope-manifold-data`
- Custom domain: `https://manifolds-data.aionoscope.langotime.ai`
- Zone: `langotime.ai`
- Zone ID: `684fb9957f1f6c827d640eceef51019f`
- R2 `r2.dev` public access is disabled; use the custom domain only.

Current data version:

```text
manifolds/v20260603T142443Z/
```

Current mutable pointer:

```text
https://manifolds-data.aionoscope.langotime.ai/manifolds/latest.json
```

Current viewer:

```text
results/manifolds.html
```

The old generated location `results/manifolds/index.html` is ignored and should
not be relied on for Pages deployment.

## Data Layout

The R2 object keys preserve the local relative paths under `results/manifolds/`
inside a versioned data prefix:

```text
manifolds/v20260603T142443Z/manifest.json
manifolds/v20260603T142443Z/<model>/<target>/metrics.json
manifolds/v20260603T142443Z/<model>/ckpt_<step>/<target>/metrics.json
manifolds/v20260603T142443Z/<model>/<target>/plots/<model>__<target>__layer_<n>_plot_data.json
manifolds/v20260603T142443Z/<model>/<target>/plots/<model>__<target>__layer_<n>_distance_data.json
```

`manifest.json` uses `schema_version = "manifold_viewer_manifest_v1"` and is
the small browser bootstrap index. It stores model/target/layer identity,
optional checkpoint identity for checkpoint sweeps, normalized per-layer plot
paths, and each target's `metrics_json` path. It does not duplicate scalar layer
metrics.

Manifold plot payloads are intentionally split:

- `*_plot_data.json`: centroid/path payload and a `distance_data_json` pointer.
- `*_distance_data.json`: distance scatter and heatmap matrices.

The viewer loads `manifest.json` first, fetches selected model/target
`metrics.json` files for scalar metrics, loads `*_plot_data.json` immediately
for selected panels, and loads `*_distance_data.json` only when the distance
block is opened.

All R2 JSON objects are uploaded as gzip-compressed bytes while keeping their
`.json` object keys. Large per-target and plot payloads are long-lived and use:

```text
Content-Type: application/json
Content-Encoding: gzip
Cache-Control: public, max-age=31536000, immutable
```

Browsers still use `response.json()` normally because the `Content-Encoding`
header tells them to decompress automatically.

`manifolds/v.../manifest.json` is the bootstrap index and is the only versioned
JSON payload with a short cache lifetime. Upload it compressed with:

```text
Content-Type: application/json
Content-Encoding: gzip
Cache-Control: public, max-age=86400
```

Do not mark `manifest.json` as `immutable`; repeat visitors should see a
refreshed index within one day.

`manifolds/latest.json` is uploaded uncompressed with:

```text
Content-Type: application/json
Cache-Control: public, max-age=60
```

It is a pointer, not the cacheable data payload.

## Viewer Behavior

`results/manifolds.html` is a checked-in shell. It does not embed manifold
records or scalar metrics. It resolves `manifest.json`, `metrics.json`, and
plot artifact paths through an optional browser global:

```js
window.MANIFOLD_DATA_BASE_URL
```

If that global is set before the viewer script runs, it wins. Otherwise the
viewer chooses a host-aware default:

- `localhost`, `127.0.0.1`, `::1`, and `file:` use local relative base
  `manifolds/` for `results/manifolds.html`, and the current directory for
  generated per-run `index.html` viewers;
- all other hosts use the R2 custom-domain base:

```text
https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/
```

Relative paths in manifest and metric metadata, such as:

```text
Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_plot_data.json
```

therefore resolve to `results/manifolds/...` when serving
`results/manifolds.html` locally, and to R2 when serving the same checked-in HTML
from Cloudflare Pages. Local development should not depend on R2 CORS.

## CORS

R2 CORS is configured on bucket `aionoscope-manifold-data`:

```text
Allowed origins: https://aionoscope-benchmarks.pages.dev, https://aionoscope.langotime.ai
Allowed methods: GET, HEAD
Allowed headers: *
Exposed headers: Content-Length, Content-Encoding, Cache-Control, cf-cache-status
Max age: 86400
```

Check it with:

```bash
npx --yes wrangler r2 bucket cors list aionoscope-manifold-data
```

## Cache Rule

The cache rule is configured in the Cloudflare zone `langotime.ai`, not in R2.
It applies only to versioned data prefixes:

```text
(http.host eq "manifolds-data.aionoscope.langotime.ai" and starts_with(http.request.uri.path, "/manifolds/v"))
```

The rule should make matching traffic eligible for cache and let the edge TTL
come from the object `Cache-Control` header. `latest.json` intentionally does
not match this rule.

Expected behavior:

- first request to a versioned JSON object: `cf-cache-status: MISS`
- subsequent request from the same edge: `cf-cache-status: HIT`
- `latest.json`: short browser cache and typically `cf-cache-status: DYNAMIC`

If the Cache Rules UI warns that DNS may not be proxied, make sure the full
expression is in the expression editor. Do not create a manual DNS record for
the R2 custom domain; the R2 custom domain already routes through Cloudflare.

## Upload Procedure

Preconditions:

- Run `npx --yes wrangler whoami` and confirm the expected Cloudflare account.
- Confirm no PNGs exist. PNG manifold artifacts are deprecated and must not be
  uploaded.
- Build or refresh the viewer manifest before uploading JSON objects:

```bash
uv run python scripts/build_manifold_calibration_viewer.py \
  --artifact-root results/manifolds \
  --manifest-only
```

```bash
find results/manifolds -type f -name '*.png' | wc -l
```

Upload every `*.json` under `results/manifolds/` to a new versioned prefix.
Use `--remote`; without it Wrangler writes to local R2 storage. The root
`results/manifolds/manifest.json` must use one-day cache metadata:

```bash
gzip -n -6 -c results/manifolds/manifest.json \
  | npx --yes wrangler r2 object put \
      aionoscope-manifold-data/manifolds/vYYYYMMDDTHHMMSSZ/manifest.json \
      --pipe \
      --content-type application/json \
      --content-encoding gzip \
      --cache-control 'public, max-age=86400' \
      --remote \
      --force
```

The checked-in Pages shell may append a short manifest-only query string when a
same-prefix manifest refresh is required. That query must not be added to
`metrics.json`, plot data, or distance data paths; those remain immutable
versioned object keys.

Example for one long-lived payload object:

```bash
gzip -n -6 -c results/manifolds/Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_plot_data.json \
  | npx --yes wrangler r2 object put \
      aionoscope-manifold-data/manifolds/vYYYYMMDDTHHMMSSZ/Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_plot_data.json \
      --pipe \
      --content-type application/json \
      --content-encoding gzip \
      --cache-control 'public, max-age=31536000, immutable' \
      --remote \
      --force
```

For bulk upload, keep concurrency conservative. High concurrency can hit
Cloudflare API `429 Too Many Requests`; `4` workers was stable for the
2026-06-03 upload, while `12` workers produced transient 429s.

After the versioned upload succeeds, update:

```text
manifolds/latest.json
```

with the new version, prefix, base URL, file count, and byte count.

## Validation

Verify the mutable pointer:

```bash
curl -sS https://manifolds-data.aionoscope.langotime.ai/manifolds/latest.json | jq .
```

Verify the viewer manifest:

```bash
curl -sS https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/manifest.json \
  | jq '.schema_version, (.records | length), .records[0].metrics_json'
```

Verify a versioned split pair:

```bash
curl -sSI -H 'Origin: https://aionoscope-benchmarks.pages.dev' \
  https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_plot_data.json

curl -sSI -H 'Origin: https://aionoscope-benchmarks.pages.dev' \
  https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_distance_data.json
```

Expected headers for long-lived payload objects:

```text
HTTP/2 200
content-type: application/json
cache-control: public, max-age=31536000, immutable
access-control-allow-origin: https://aionoscope-benchmarks.pages.dev
```

Expected headers for the versioned manifest:

```text
HTTP/2 200
content-type: application/json
cache-control: public, max-age=86400
access-control-allow-origin: https://aionoscope-benchmarks.pages.dev
```

Run repeated `HEAD` requests against a versioned JSON object to confirm edge
cache:

```bash
for i in 1 2 3; do
  curl -sSI https://manifolds-data.aionoscope.langotime.ai/manifolds/v20260603T142443Z/Chronos-2/gaussian_time_frac/plots/Chronos-2__gaussian_time_frac__layer_0_plot_data.json \
    | rg -i 'HTTP/|cf-cache-status|age:|cache-control'
  sleep 1
done
```

`wrangler r2 bucket info` object counts can lag after large uploads. Do not use
that aggregate alone as the source of truth for upload completeness; compare the
local JSON list to the upload success log and sample public URLs.

## Why Not Pages-Only Or D1

Pages-only is wrong for the manifold JSON corpus because the generated data is
large, changes independently of the viewer shell, and includes many megabyte-size
objects. Cloudflare Pages should receive the HTML shell from Git, not tens of GB
of generated data.

D1/KV are also wrong for the plot payloads: these are static object blobs that
the browser fetches by path. R2 plus CDN caching is the appropriate storage and
delivery layer.
