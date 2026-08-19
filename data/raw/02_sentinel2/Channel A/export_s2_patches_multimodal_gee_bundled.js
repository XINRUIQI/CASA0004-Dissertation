// ============================================================================
// AOI config for GEE scripts — AUTO-GENERATED, do not edit by hand.
// Source : aoi_oil_infrastructure.csv
// Refresh: python sync_aoi_csv_to_gee.py
// ============================================================================
//
// GEE cannot read local CSV at runtime. Either:
//   A) Use this embedded FeatureCollection (default, USE_EE_ASSET = false), or
//   B) Upload aoi_oil_infrastructure.csv to GEE Assets, set USE_EE_ASSET = true
//      and AOI_ASSET_ID below, then re-run this script (embedded FC still kept
//      as fallback documentation).
//
// ============================================================================

var USE_EE_ASSET = false;
var AOI_ASSET_ID = 'projects/YOUR_PROJECT/assets/aoi_oil_infrastructure';

var PATCH_HALF_BY_TYPE = {
  "port": 3200,
  "refinery": 2560,
  "terminal": 1280
};

var AOI_SITES_EMBEDDED = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point([4.145, 51.95]), {site_id: "P001", short_name: "Rotterdam", site_name: "Port of Rotterdam", site_type: "port", country: "Netherlands", region: "Europe", lon: 4.145, lat: 51.95, buffer_km: 5, patch_half_m: 3200}),
  ee.Feature(ee.Geometry.Point([56.356, 25.199]), {site_id: "P002", short_name: "Fujairah", site_name: "Fujairah Oil Terminal", site_type: "terminal", country: "United Arab Emirates", region: "Middle East", lon: 56.356, lat: 25.199, buffer_km: 5, patch_half_m: 1600}),
  ee.Feature(ee.Geometry.Point([50.157, 26.643]), {site_id: "P003", short_name: "RasTanura", site_name: "Ras Tanura Terminal", site_type: "terminal", country: "Saudi Arabia", region: "Middle East", lon: 50.157, lat: 26.643, buffer_km: 5, patch_half_m: 1280}),
  ee.Feature(ee.Geometry.Point([103.708, 1.274]), {site_id: "P004", short_name: "Jurong", site_name: "Singapore Jurong Island", site_type: "refinery", country: "Singapore", region: "Asia", lon: 103.708, lat: 1.274, buffer_km: 5, patch_half_m: 2560}),
  ee.Feature(ee.Geometry.Point([-95.1, 29.736]), {site_id: "P005", short_name: "Houston", site_name: "Houston Ship Channel", site_type: "port", country: "United States", region: "North America", lon: -95.1, lat: 29.736, buffer_km: 5, patch_half_m: 3200}),
  ee.Feature(ee.Geometry.Point([121.982, 29.935]), {site_id: "P006", short_name: "NingboZhoushan", site_name: "Ningbo-Zhoushan Port", site_type: "port", country: "China", region: "East Asia", lon: 121.982, lat: 29.935, buffer_km: 5, patch_half_m: 3200}),
  ee.Feature(ee.Geometry.Point([69.86, 22.345]), {site_id: "P007", short_name: "Jamnagar", site_name: "Jamnagar Refinery", site_type: "refinery", country: "India", region: "South Asia", lon: 69.86, lat: 22.345, buffer_km: 5, patch_half_m: 2560}),
  ee.Feature(ee.Geometry.Point([48.81, 29.681]), {site_id: "P008", short_name: "Basra", site_name: "Basra Oil Terminal", site_type: "terminal", country: "Iraq", region: "Middle East", lon: 48.81, lat: 29.681, buffer_km: 5, patch_half_m: 800}),
  ee.Feature(ee.Geometry.Point([129.343, 35.433]), {site_id: "P009", short_name: "Ulsan", site_name: "Ulsan Refinery", site_type: "refinery", country: "South Korea", region: "East Asia", lon: 129.343, lat: 35.433, buffer_km: 5, patch_half_m: 2560}),
  ee.Feature(ee.Geometry.Point([50.324, 29.231]), {site_id: "P010", short_name: "Kharg", site_name: "Kharg Island Terminal", site_type: "terminal", country: "Iran", region: "Middle East", lon: 50.324, lat: 29.231, buffer_km: 5, patch_half_m: 1600}),
  ee.Feature(ee.Geometry.Point([38.229, 23.961]), {site_id: "P011", short_name: "Yanbu", site_name: "Yanbu Export Terminal", site_type: "terminal", country: "Saudi Arabia", region: "Middle East", lon: 38.229, lat: 23.961, buffer_km: 5, patch_half_m: 1600})
]);

function aoiSitesRaw() {
  if (USE_EE_ASSET) {
    return ee.FeatureCollection(AOI_ASSET_ID);
  }
  return AOI_SITES_EMBEDDED;
}

// Channel B: circular buffer around centre (buffer_km from CSV).
function aoiSitesBuffered() {
  return aoiSitesRaw().map(function(f) {
    var bufferM = ee.Number(f.get('buffer_km')).multiply(1000);
    return f.setGeometry(
      ee.Geometry.Point([f.get('lon'), f.get('lat')]).buffer(bufferM)
    );
  });
}

// Channel A export: client-side site list for forEach + getInfo loops.
function aoiSitesClient() {
  return aoiSitesRaw().getInfo().features.map(function(f) {
    var p = f.properties;
    return {
      id: p.site_id,
      name: p.short_name,
      type: p.site_type,
      lon: p.lon,
      lat: p.lat,
      patch_half_m: p.patch_half_m,
      site_name: p.site_name,
      country: p.country,
      region: p.region
    };
  });
}

function patchHalfM(site) {
  return site.patch_half_m ||
    PATCH_HALF_BY_TYPE[site.type] ||
    PATCH_HALF_BY_TYPE[site.site_type] ||
    1280;
}


// --- script body (from export_s2_patches_multimodal_gee.js) ---

// ============================================================================
// Sentinel-2 Multi-AOI Patch Export for End-to-End Multimodal Oil-Price Model
// ----------------------------------------------------------------------------
// Purpose : Export monthly, cloud-masked, foundation-model-ready S2 patches
//           for all 11 oil-infrastructure AOIs, to feed a frozen EO encoder
//           (e.g. Prithvi-EO-2.0 / SatMAE) instead of tabular indices only.
//
// Output  : One GeoTIFF per (site, month) with 6 reflectance bands +
//           one manifest CSV (n valid scenes, min cloud %) per (site, month).
//
// Bands   : B2,B3,B4,B8A,B11,B12  (Blue,Green,Red,NIR-narrow,SWIR1,SWIR2)
//           -> matches the HLS 6-band set used by Prithvi-EO foundation models.
// Cloud   : SCL-based masking, then monthly median composite. One getInfo per
//           site (scene counts for all months at once, never per-image) lets
//           us skip empty months instead of queuing failing 0-band exports.
// Window  : 2019-01 .. 2026-06  (aligned with the standardised M0-M4 window).
// Patch   : per-site half-size from aoi_oil_infrastructure.csv (patch_half_m).
// CRS     : per-site UTM (true metric pixels), computed from longitude/latitude.
//
// PREREQUISITE: load_aoi_config_gee.js in the same GEE project (or paste above).
// Regenerate from CSV: python sync_aoi_csv_to_gee.py
// ============================================================================

// ----------------------------- CONFIG ---------------------------------------
var START_YEAR  = 2019, START_MONTH = 1;   // inclusive
var END_YEAR    = 2026, END_MONTH   = 6;    // inclusive
var SCALE        = 10;                        // metres / pixel
var CLOUD_MAX    = 60;                        // pre-filter scenes by scene cloud %
var BANDS        = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'];
var EXPORT_FOLDER = 'CASA0004_S2_patches';   // Google Drive folder

// Run one site at a time by setting e.g. 'P001', or 'ALL' to queue every site.
var RUN_SITE_ID = 'ALL';

// AOI sites + patch_half_m from aoi_oil_infrastructure.csv (via load_aoi_config_gee.js)
var AOIS = aoiSitesClient();

// ----------------------------- HELPERS --------------------------------------

// UTM EPSG code from lon/lat (northern -> 326xx, southern -> 327xx).
function utmEpsg(lon, lat) {
  var zone = Math.floor((lon + 180) / 6) + 1;
  var base = (lat >= 0) ? 32600 : 32700;
  return 'EPSG:' + (base + zone);
}

// SCL-based cloud / shadow / cirrus / snow mask.
function maskS2(img) {
  var scl = img.select('SCL');
  var bad = scl.eq(3)   // cloud shadow
    .or(scl.eq(8))      // cloud medium prob
    .or(scl.eq(9))      // cloud high prob
    .or(scl.eq(10))     // thin cirrus
    .or(scl.eq(11));    // snow / ice
  return img.updateMask(bad.not());
}

// Zero-padded YYYY_MM label.
function monthLabel(y, m) {
  var mm = (m < 10) ? ('0' + m) : ('' + m);
  return y + '_' + mm;
}

// Build the client-side list of {year, month, label, start, end}.
function buildMonths() {
  var out = [];
  var y = START_YEAR, m = START_MONTH;
  while (y < END_YEAR || (y === END_YEAR && m <= END_MONTH)) {
    var start = ee.Date.fromYMD(y, m, 1);
    var end   = start.advance(1, 'month');
    out.push({year: y, month: m, label: monthLabel(y, m), start: start, end: end});
    m += 1;
    if (m > 12) { m = 1; y += 1; }
  }
  return out;
}

// ----------------------------- MAIN -----------------------------------------

var S2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var MONTHS = buildMonths();
print('Months to export per site:', MONTHS.length);
print('AOI source:', USE_EE_ASSET ? AOI_ASSET_ID : 'embedded CSV sync');

var sitesToRun = AOIS.filter(function(s) {
  return (RUN_SITE_ID === 'ALL') || (s.id === RUN_SITE_ID);
});
print('Sites to run:', sitesToRun.map(function(s){return s.id;}));

var manifestFeatures = [];  // manifest rows (one per site-month, incl. empties)
var taskCount   = 0;        // image export tasks actually queued
var skippedCount = 0;       // site-months skipped because no scene is available

sitesToRun.forEach(function(site) {
  var halfM  = patchHalfM(site);
  var center = ee.Geometry.Point([site.lon, site.lat]);
  var region = center.buffer(halfM).bounds();
  var crs    = utmEpsg(site.lon, site.lat);

  // Per-month filtered collections for this site.
  var monthlies = MONTHS.map(function(mo) {
    return S2
      .filterBounds(center)
      .filterDate(mo.start, mo.end)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_MAX));
  });

  // ONE getInfo per site (not per image): resolve the scene count for every
  // month at once so we can (a) skip empty months on export and (b) write
  // plain-number manifest rows with no risk of null properties.
  var sceneCounts = ee.List(monthlies.map(function(c) { return c.size(); })).getInfo();

  MONTHS.forEach(function(mo, i) {
    var monthly = monthlies[i];
    var n       = sceneCounts[i];

    // Empty month: no scene passes the cloud pre-filter. A median over an
    // empty collection yields a 0-band image and the export task fails, so
    // skip queuing it. The manifest still records the month with n_scenes = 0.
    if (n > 0) {
      // Cloud-masked median composite for the month, 6 bands, Int16.
      var composite = monthly
        .map(maskS2)
        .select(BANDS)
        .median()
        .toInt16()
        .clip(region);

      var desc = 'S2_' + site.id + '_' + site.name + '_' + mo.label;

      Export.image.toDrive({
        image: composite,
        description: desc,
        folder: EXPORT_FOLDER,
        fileNamePrefix: desc,
        region: region,
        scale: SCALE,
        crs: crs,
        maxPixels: 1e8,
        fileFormat: 'GeoTIFF'
      });
      taskCount += 1;
    } else {
      skippedCount += 1;
    }

    // Manifest row. n_scenes is a resolved client number; cloud stats are only
    // meaningful (and non-null) when the month has scenes, else a -1 sentinel.
    manifestFeatures.push(ee.Feature(null, {
      site_id:   site.id,
      site_name: site.name,
      site_type: site.type,
      month:     mo.label,
      year:      mo.year,
      lon:       site.lon,
      lat:       site.lat,
      crs:       crs,
      patch_half_m: halfM,
      patch_px:     (2 * halfM) / SCALE,
      n_scenes:  n,
      exported:  (n > 0) ? 1 : 0,
      min_cloud:  (n > 0) ? monthly.aggregate_min('CLOUDY_PIXEL_PERCENTAGE') : -1,
      mean_cloud: (n > 0) ? monthly.aggregate_mean('CLOUDY_PIXEL_PERCENTAGE') : -1
    }));
  });
});

print('Total image export tasks queued:', taskCount);
print('Site-months skipped (no scenes):', skippedCount);

// Single manifest CSV for all queued (site, month) cells.
var manifest = ee.FeatureCollection(manifestFeatures);
Export.table.toDrive({
  collection: manifest,
  description: 'S2_patches_manifest_' + RUN_SITE_ID,
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'S2_patches_manifest_' + RUN_SITE_ID,
  fileFormat: 'CSV'
});

// ----------------------------- PREVIEW --------------------------------------
// Quick visual check on the first site, most recent clear scene.
var s0 = sitesToRun[0];
var half0 = patchHalfM(s0);
var c0 = ee.Geometry.Point([s0.lon, s0.lat]);
var preview = S2.filterBounds(c0)
  .filterDate(START_YEAR + '-01-01', END_YEAR + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .sort('system:time_start', false)
  .first();
Map.centerObject(c0, 13);
Map.addLayer(preview, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, s0.id + ' RGB');
Map.addLayer(c0.buffer(half0).bounds(), {color: 'red'}, 'Patch region (' + s0.type + ')');
