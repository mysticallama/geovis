# Troubleshooting Planet Imagery Downloads

## Common Download Errors and Solutions

### Error: "Could not get download URL for [item_id]/[asset_type]"

This error occurs when asset activation fails or times out. Here are the main causes and solutions:

---

## 1. Asset Activation Timeout

**Symptom:** The download gets stuck waiting for activation, then times out after 10 minutes.

**Cause:** Planet's servers are slow to prepare the asset, or the asset is in a queue.

**Solutions:**

### Increase Timeout
The default timeout is now 600 seconds (10 minutes). To increase it:

```python
from planet_pipeline import PlanetPipeline

pipeline = PlanetPipeline(storage_dir="./data")

# Increase activation timeout to 20 minutes
pipeline.downloader.client.get_download_url(asset_url, timeout=1200)
```

### Check Asset Status Manually
Use Planet's API to check the asset status:

```python
# Get asset info
assets = pipeline.client.get_item_assets("PSScene", "20241219_190818_77_24e1")
asset = assets["ortho_analytic_4b"]

print(f"Status: {asset['status']}")
print(f"Type: {asset['type']}")
print(f"Permissions: {asset.get('permissions', 'N/A')}")
```

---

## 2. Missing Permissions / Subscription Required

**Symptom:** Error message includes "Access denied" or asset status remains "inactive".

**Cause:** Your Planet account doesn't have permission to download this asset type or scene.

**Solutions:**

### Check Your Subscription
Different Planet subscriptions have access to different asset types:

- **Basic Access:** Usually includes `ortho_visual`
- **Education/Research:** May include `ortho_analytic_4b`
- **Commercial:** Full access to all asset types

**What you can try:**

1. **Use a different asset type:**
```python
# Instead of ortho_analytic_4b, try:
pipeline.download_imagery(asset_types=["ortho_visual"])
```

2. **Check asset availability:**
```python
assets = pipeline.client.get_item_assets("PSScene", "scene_id")
print("Available assets:", list(assets.keys()))

# Download only what's available
available = [k for k, v in assets.items() if v.get('status') != 'inactive']
pipeline.download_imagery(asset_types=available)
```

3. **Contact Planet Support:**
- Email: support@planet.com
- Request access to specific asset types or scenes

---

## 3. Rate Limiting (HTTP 429 Error)

**Symptom:** Error messages about "Too Many Requests" or "Rate limited".

**Cause:** Making too many API requests too quickly.

**Solution:** The pipeline now handles this automatically with:
- Exponential backoff (1s → 2s → 4s → 8s → 16s)
- Reduced parallel downloads (2 instead of 4)
- Automatic retries (up to 5 attempts)

If you still hit rate limits:

```python
# Reduce to 1 parallel download
pipeline.downloader.max_workers = 1

# Increase delay between requests
pipeline.downloader.rate_limit_delay = 2.0  # 2 seconds base delay

# Limit downloads per batch
pipeline.download_imagery(limit_per_aoi=10)
```

---

## 4. Scene Not Available / Expired

**Symptom:** Asset is listed in search results but won't download.

**Cause:** The scene may have expired or been removed from Planet's archive.

**Solutions:**

1. **Filter by asset availability during search:**
```python
# Only include scenes with ortho_analytic_4b available
results = pipeline.query_all_aois(
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Filter results
available_scenes = []
for scene in results["my_aoi"]:
    item_id = scene["id"]
    item_type = scene["properties"]["item_type"]

    try:
        assets = pipeline.client.get_item_assets(item_type, item_id)
        if "ortho_analytic_4b" in assets:
            available_scenes.append(scene)
    except:
        pass

print(f"Available: {len(available_scenes)} out of {len(results['my_aoi'])}")
```

2. **Use skip_existing to skip failed downloads:**
```python
# Will skip scenes that already failed
pipeline.download_imagery(skip_existing=True)
```

---

## 5. Network Issues

**Symptom:** Connection errors, timeouts, or incomplete downloads.

**Solutions:**

1. **Check your internet connection**

2. **Increase chunk size for large files:**
```python
pipeline.downloader.chunk_size = 16384  # 16KB chunks (default is 8KB)
```

3. **Retry failed downloads:**
The pipeline automatically retries on network errors. Check logs for details.

---

## Diagnostic Commands

### Check Scene and Asset Details

```python
from planet_pipeline import PlanetPipeline

pipeline = PlanetPipeline()

# Get scene info
item_id = "20241219_190818_77_24e1"
item_type = "PSScene"

# Get all assets for this scene
assets = pipeline.client.get_item_assets(item_type, item_id)

print(f"\nScene: {item_id}")
print(f"Available assets:")
for asset_name, asset_info in assets.items():
    status = asset_info.get('status', 'unknown')
    print(f"  - {asset_name}: {status}")

    if status == 'inactive':
        print(f"    Reason: {asset_info.get('error', 'May require activation or permissions')}")
```

### Check Your Download Log

The pipeline logs everything. Look for patterns:

```bash
# Search logs for errors
grep "ERROR" planet_pipeline.log

# Check specific scene
grep "20241219_190818_77_24e1" planet_pipeline.log

# See all activation attempts
grep "Activating asset" planet_pipeline.log
```

### Test Single Scene Download

```python
# Test downloading just one scene
pipeline.downloader.download_by_ids(
    item_ids=["20241219_190818_77_24e1"],
    item_type="PSScene",
    aoi_name="test",
    asset_types=["ortho_analytic_4b"]
)
```

---

## Best Practices to Avoid Download Issues

### 1. Query with Realistic Filters
```python
# Be conservative with cloud cover and date range
results = pipeline.query_all_aois(
    start_date="2024-01-01",
    end_date="2024-01-31",  # Shorter range
    cloud_cover_max=0.1,     # Lower threshold
    item_types=["PSScene"]
)
```

### 2. Limit Downloads During Testing
```python
# Test with small batches first
pipeline.download_imagery(limit_per_aoi=5)
```

### 3. Use Available Asset Types
```python
# Check what's available for your subscription
# Common alternatives:
# - ortho_visual (RGB, most accessible)
# - ortho_analytic_sr (surface reflectance)
# - ortho_analytic_8b (8-band, if available)
```

### 4. Monitor Progress
```python
import logging

# Enable debug logging to see detailed progress
logging.basicConfig(level=logging.DEBUG)
```

### 5. Handle Failures Gracefully
```python
# The pipeline already does this, but you can add custom handling
downloaded_files = pipeline.download_imagery()

# Check what succeeded
print(f"Successfully downloaded: {len(downloaded_files['my_aoi'])} files")
```

---

## Getting Help

If you're still stuck:

1. **Check Planet's Status Page:** https://status.planet.com/
2. **Review Planet's API Docs:** https://developers.planet.com/
3. **Contact Planet Support:** support@planet.com
4. **Check the logs:** Look for specific error messages

### Sharing Error Information

When asking for help, include:
- The complete error message
- Item ID and asset type
- Your Planet subscription type
- Relevant log excerpts

---

## Recent Fixes in This Pipeline

✅ **Fixed:** Asset activation now uses POST instead of GET (required by Planet API)
✅ **Added:** Better error messages for missing permissions
✅ **Added:** Increased default timeout from 5 to 10 minutes
✅ **Added:** Status logging during activation
✅ **Added:** Automatic retry with exponential backoff for rate limits
✅ **Added:** HTTP 403 detection for permission errors

The pipeline should now handle most common download issues automatically!
