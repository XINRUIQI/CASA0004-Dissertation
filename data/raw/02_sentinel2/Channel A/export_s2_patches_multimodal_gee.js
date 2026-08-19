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
// Patch   : 2.56 km x 2.56 km centred on each AOI -> 256 x 256 px @ 10 m.
// CRS     : per-site UTM (true metric pixels), computed from longitude/latitude.
// ============================================================================

// ----------------------------- CONFIG ---------------------------------------
var START_YEAR  = 2019, START_MONTH = 1;   // inclusive
var END_YEAR    = 2026, END_MONTH   = 6;    // inclusive
var SCALE        = 10;                        // metres / pixel
var CLOUD_MAX    = 60;                        // pre-filter scenes by scene cloud %
var BANDS        = ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'];
var EXPORT_FOLDER = 'CASA0004_S2_patches';   // Google Drive folder

// Differentiated patch HALF-size (metres) by infrastructure type.
// Full patch = 2 x half; pixels @10 m = full / 10. The frozen EO encoder
// resizes to a fixed input later, so different px sizes per type are fine.
//   port      -> 6.4 km (640 px)  sprawling port/industrial corridors
//   refinery  -> 5.12 km (512 px) large refining / petrochemical complexes
//   terminal  -> 2.56 km (256 px) compact export / storage terminals
var PATCH_HALF_BY_TYPE = {
  port:     3200,
  refinery: 2560,
  terminal: 1280
};

// Per-SITE half-size override (metres), takes precedence over PATCH_HALF_BY_TYPE.
// Used when a specific AOI needs a wider/narrower window than its type default
// after visual inspection of the framing.
//   P002 Fujairah -> 3.2 km (320 px): storage / anchorage cluster spills past 2.56 km.
//   P008 Basra    -> 1.6 km (160 px): compact riverside berths, tighter framing.
//   P010 Kharg    -> 3.2 km (320 px): island terminal plus offshore loading jetties.
//   P011 Yanbu    -> 3.2 km (320 px): export berths plus tank-farm spread.
var PATCH_HALF_BY_SITE = {
  P002: 1600,
  P008:  800,
  P010: 1600,
  P011: 1600
};

// Run one site at a time by setting e.g. 'P001', or 'ALL' to queue every site.
var RUN_SITE_ID = 'ALL';

// ----------------------------- AOI SOURCE -----------------------------------
// Coordinates / type / name come from aoi_oil_infrastructure.csv, uploaded to
// GEE as a table asset (GEE cannot read a local CSV). Set AOI_ASSET to that
// asset's ID; ingest lon -> longitude, lat -> latitude so each row gets a point.
// One small getInfo (11 rows) pulls the table client-side so the per-site
// Export loop below can stay plain JavaScript.
var AOI_ASSET = 'projects/ee-USERNAME/assets/aoi_oil_infrastructure';

var AOIS = ee.FeatureCollection(AOI_ASSET).getInfo().features
  .map(function(ft) {
    var p = ft.properties;
    return {
      id:   p.site_id,
      name: p.site_short,   // filename-safe short name (no spaces)
      type: p.site_type,
      lon:  p.lon,
      lat:  p.lat,
      // Coordinate provenance (carried through to the manifest).
      reference_name:  p.reference_name,
      reference_lon:   p.reference_lon,
      reference_lat:   p.reference_lat,
      coord_offset_km: p.coord_offset_km,
      coord_method:    p.coord_method,
      coord_checked:   p.coord_checked
    };
  })
  .sort(function(a, b) { return a.id < b.id ? -1 : (a.id > b.id ? 1 : 0); });

// ----------------------------- HELPERS --------------------------------------

// UTM EPSG code from lon/lat (northern -> 326xx, southern -> 327xx).
function utmEpsg(lon, lat) {
  var zone = Math.floor((lon + 180) / 6) + 1;
  var base = (lat >= 0) ? 32600 : 32700;
  return 'EPSG:' + (base + zone);
}

// SCL-based cloud / shadow / cirrus / snow mask.
// Keep: 4 veg, 5 bare, 6 water, 7 unclassified, 2 dark, 1 saturated->drop? keep simple.
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

var sitesToRun = AOIS.filter(function(s) {
  return (RUN_SITE_ID === 'ALL') || (s.id === RUN_SITE_ID);
});
print('Sites to run:', sitesToRun.map(function(s){return s.id;}));

var manifestFeatures = [];  // manifest rows (one per site-month, incl. empties)
var taskCount   = 0;        // image export tasks actually queued
var skippedCount = 0;       // site-months skipped because no scene is available

sitesToRun.forEach(function(site) {
  var halfM  = PATCH_HALF_BY_SITE[site.id] || PATCH_HALF_BY_TYPE[site.type] || 1280;
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
      reference_lon:   site.reference_lon,
      reference_lat:   site.reference_lat,
      reference_name:  site.reference_name,
      coord_offset_km: site.coord_offset_km,
      coord_method:    site.coord_method,
      coord_order:     'longitude_latitude',
      coord_checked:   site.coord_checked,
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
var half0 = PATCH_HALF_BY_SITE[s0.id] || PATCH_HALF_BY_TYPE[s0.type] || 1280;
var c0 = ee.Geometry.Point([s0.lon, s0.lat]);
var preview = S2.filterBounds(c0)
  .filterDate(START_YEAR + '-01-01', END_YEAR + '-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .sort('system:time_start', false)
  .first();
Map.centerObject(c0, 13);
Map.addLayer(preview, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, s0.id + ' RGB');
Map.addLayer(c0.buffer(half0).bounds(), {color: 'red'}, 'Patch region (' + s0.type + ')');
