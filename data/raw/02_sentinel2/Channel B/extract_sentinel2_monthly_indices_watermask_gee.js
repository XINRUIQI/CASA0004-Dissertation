// =======================================================
// Sentinel-2 Monthly Oil Infrastructure Indicators — WATER-MASK VERSION
//
// DIFFERENCE FROM ORIGINAL (extract_sentinel2_monthly_indices_gee.js):
//
//   ① MNDWI water mask applied per image BEFORE compositing.
//      MNDWI = (Green − SWIR1) / (Green + SWIR1)
//            = (B3 − B11) / (B3 + B11)           [Xu 2006]
//      Pixels where MNDWI > WATER_THRESH are classified as water and
//      MASKED OUT from NDVI / NDBI / BSI calculation.
//      NDWI is kept unmasked (it IS a water-surface index).
//
//   ② MNDWI exported as extra column (diagnostic; Xu 2006).
//
//   ③ is_water band (per-image float 0/1) added per image.
//      Monthly mean of is_water ≈ water-pixel fraction in AOI.
//      land_pixel_fraction = 1 − is_water (set in .map() below).
//
//   ④ Memory optimisations vs first draft:
//      - is_water incorporated into the SAME two reduceRegions calls as the
//        original script (no third separate call → original memory budget).
//      - Map.addLayer removed (triggers eager evaluation on large collections).
//      - tileScale raised from 4 to 8.
//
// RATIONALE (B4 robustness, channelB_mechanism_plan.md §3 B4):
//   B0 flagged high NDWI variance at water-dominated terminals (Basra, Kharg,
//   Jurong …). SHAP results (2026-06-23) put NDWI_anom_Kharg and
//   NDWI_anom_Yanbu as top-2 M2 features, suggesting partial tidal/turbidity
//   confound. Masking water pixels isolates land-activity signal for NDVI/NDBI/BSI.
//
// REFERENCES:
//   McFeeters 1996 IJRS 17(7): 1425–1432  (NDWI  = B3−B8  / B3+B8)
//   Xu 2006       IJRS 27(14): 3025–3033  (MNDWI = B3−B11 / B3+B11)
//
// Output: monthly NDVI(land), NDWI(all), NDBI(land), BSI(land),
//         MNDWI(all), land_pixel_fraction per AOI
// Period: 2017-04 ~ 2025-12  |  AOIs: 11 global oil infrastructure sites
// =======================================================


// -------------------------------
// 1. User settings
// -------------------------------

var START_DATE = '2017-04-01';
var END_DATE   = '2025-12-31';

var CLOUD_FILTER   = 60;   // pre-filter: max scene cloud %  (same as original)
var CLD_PRB_THRESH = 40;   // s2cloudless probability threshold (same as original)

// MNDWI > WATER_THRESH → water pixel (Xu 2006 recommends 0; use 0 here)
var WATER_THRESH = 0.0;

var EXPORT_FOLDER = 'CASA0004_oil_project';


// -------------------------------
// 2. AOI table  (same as original)
// -------------------------------

var AOI_ASSET = 'projects/ee-USERNAME/assets/aoi_oil_infrastructure';

var sites = ee.FeatureCollection(AOI_ASSET)
  .select(['site_id', 'site_name', 'site_type', 'country', 'region', 'buffer_km'])
  .map(function(f) {
    return f.buffer(ee.Number(f.get('buffer_km')).multiply(1000));
  });

// NOTE: Map.centerObject and Map.addLayer are intentionally omitted.
// Visualising large ImageCollections in the map panel triggers eager
// evaluation and causes "User memory limit exceeded" during Export/print.


// -------------------------------
// 3. Sentinel-2 collections  (same as original)
// -------------------------------

var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(sites)
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER));

var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterDate(START_DATE, END_DATE)
  .filterBounds(sites);

var join = ee.Join.saveFirst('s2cloudless');

var joined = ee.ImageCollection(join.apply({
  primary:   s2Sr,
  secondary: s2Clouds,
  condition: ee.Filter.equals({
    leftField:  'system:index',
    rightField: 'system:index'
  })
}));


// -------------------------------
// 4. Cloud mask  (same as original)
// -------------------------------

function addCloudProbability(img) {
  var cloudProbImg = ee.Image(img.get('s2cloudless'));
  return img.addBands(cloudProbImg.select('probability').rename('cloud_probability'));
}

function maskClouds(img) {
  var isNotCloud = img.select('cloud_probability').lt(CLD_PRB_THRESH);
  var scl = img.select('SCL');
  var sclMask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9))
                           .and(scl.neq(10)).and(scl.neq(11));
  return img.updateMask(isNotCloud).updateMask(sclMask);
}


// -------------------------------
// 5. Spectral indices + MNDWI water mask  [KEY CHANGE]
// -------------------------------

function addIndicesWithWaterMask(img) {
  // MNDWI and NDWI: computed on ALL valid (cloud-free) pixels
  var mndwi = img.normalizedDifference(['B3', 'B11']).rename('MNDWI'); // Xu 2006
  var ndwi  = img.normalizedDifference(['B3', 'B8']).rename('NDWI');   // McFeeters 1996

  // Land mask: 1 on land/impervious/bare pixels, masked (NaN) on water
  var landMask = mndwi.lte(WATER_THRESH);

  // NDVI, NDBI, BSI: restricted to land pixels only
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI').updateMask(landMask);
  var ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI').updateMask(landMask);
  var bsi  = img.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    { SWIR: img.select('B11'), RED: img.select('B4'),
      NIR:  img.select('B8'),  BLUE: img.select('B2') }
  ).rename('BSI').updateMask(landMask);

  // is_water: float (1 = water pixel, 0 = land pixel) for ALL valid pixels
  // Used later to compute land_pixel_fraction = 1 - mean(is_water over AOI).
  var isWater = mndwi.gt(WATER_THRESH).toFloat().rename('is_water');

  return img
    .addBands([ndvi, ndwi, ndbi, bsi, mndwi, isWater])
    .copyProperties(img, ['system:time_start', 'system:index']);
}

var s2Processed = joined
  .map(addCloudProbability)
  .map(maskClouds)
  .map(addIndicesWithWaterMask)
  .select(['NDVI', 'NDWI', 'NDBI', 'BSI', 'MNDWI', 'cloud_probability', 'is_water']);


// -------------------------------
// 6. Monthly composites  [SAME STRUCTURE AS ORIGINAL — still TWO reduceRegions]
//
//    stats    : reduceRegions(mean) on monthlyMedian
//               → property names = band names (NDVI, NDWI, NDBI, BSI, MNDWI,
//                 cloud_probability, is_water)
//    statsStd : reduceRegions(mean) on monthlyStdDev image
//               → property names = band_stdDev (NDVI_stdDev, NDWI_stdDev, …)
//
//    is_water included in monthlyMedian bands → is_water mean per AOI
//    land_pixel_fraction = 1 - is_water  (set in the .map() below)
//    No third reduceRegions call needed.
// -------------------------------

var start     = ee.Date(START_DATE);
var end       = ee.Date(END_DATE);
var nMonths   = end.difference(start, 'month').round();
var monthList = ee.List.sequence(0, nMonths.subtract(1));

var monthlyFeatures = ee.FeatureCollection(
  monthList.map(function(m) {
    var startMonth    = start.advance(m, 'month');
    var endMonth      = startMonth.advance(1, 'month');
    var monthlyCol    = s2Processed.filterDate(startMonth, endMonth);
    var validObsCount = monthlyCol.size();

    // Median composite for index values
    var monthlyMedian = monthlyCol
      .select(['NDVI', 'NDWI', 'NDBI', 'BSI', 'MNDWI', 'cloud_probability', 'is_water'])
      .median();

    // StdDev image (bands named NDVI_stdDev, NDWI_stdDev, …, is_water_stdDev)
    var monthlyStdDev = monthlyCol
      .select(['NDVI', 'NDWI', 'NDBI', 'BSI', 'MNDWI'])
      .reduce(ee.Reducer.stdDev());

    // reduceRegions ①: mean of median composite per AOI
    var stats = monthlyMedian.reduceRegions({
      collection: sites,
      reducer:    ee.Reducer.mean(),
      scale:      20,
      tileScale:  8    // raised from 4 to reduce per-tile memory
    });

    // reduceRegions ②: mean of stdDev image per AOI
    var statsStd = monthlyStdDev.reduceRegions({
      collection: sites,
      reducer:    ee.Reducer.mean(),
      scale:      20,
      tileScale:  8
    });

    // Merge: join statsStd into stats by site_id (same pattern as original)
    stats = stats.map(function(f) {
      var siteId    = f.get('site_id');
      var matchStd  = statsStd.filter(ee.Filter.eq('site_id', siteId)).first();
      var isWaterMean = ee.Number(f.get('is_water'));  // from reduceRegions ①
      return f.set({
        year:                  startMonth.get('year'),
        month:                 startMonth.get('month'),
        date_month:            startMonth.format('YYYY-MM'),
        valid_obs_count:       validObsCount,
        sensor:                'Sentinel-2',
        water_mask_applied:    1,
        water_thresh_mndwi:    WATER_THRESH,
        // StdDev columns
        NDVI_std:   matchStd.get('NDVI_stdDev'),
        NDWI_std:   matchStd.get('NDWI_stdDev'),
        NDBI_std:   matchStd.get('NDBI_stdDev'),
        BSI_std:    matchStd.get('BSI_stdDev'),
        MNDWI_std:  matchStd.get('MNDWI_stdDev'),
        // Land-pixel fraction (0–1; computed from is_water mean, no third reduceRegions)
        land_pixel_fraction: ee.Number(1).subtract(isWaterMean)
      });
    });

    return stats;
  })
).flatten();


// -------------------------------
// 7. Inspect — limit to a small sample to avoid memory in console
// -------------------------------

print('Monthly S2 water-masked features (first 5 rows)',
      monthlyFeatures.limit(5));


// -------------------------------
// 8. Export to Google Drive
// -------------------------------

Export.table.toDrive({
  collection:     monthlyFeatures,
  description:    'sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi',
  folder:         EXPORT_FOLDER,
  fileNamePrefix: 'sentinel2_oil_sites_monthly_indices_watermask_201704_202512_11aoi',
  fileFormat:     'CSV'
});
