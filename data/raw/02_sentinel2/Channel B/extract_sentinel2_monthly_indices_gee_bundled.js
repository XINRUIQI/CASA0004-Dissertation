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


// --- script body (from extract_sentinel2_monthly_indices_gee.js) ---

// =======================================================
// Sentinel-2 Monthly Oil Infrastructure Indicators
// Output: monthly NDVI, NDWI, NDBI, BSI by oil-related AOI
// Dataset: COPERNICUS/S2_SR_HARMONIZED + S2 Cloud Probability
// Period: 2017-04 ~ 2025-12
// AOIs: 11 global oil infrastructure sites (aoi_oil_infrastructure.csv)
//
// PREREQUISITE: load_aoi_config_gee.js in the same GEE project (or paste above).
// Regenerate from CSV: python sync_aoi_csv_to_gee.py
// =======================================================


// -------------------------------
// 1. User settings
// -------------------------------

var START_DATE = '2017-04-01';
var END_DATE   = '2025-12-31';

var CLOUD_FILTER = 60;
var CLD_PRB_THRESH = 40;
var EXPORT_FOLDER = 'CASA0004_oil_project';


// -------------------------------
// 2. AOI table — from aoi_oil_infrastructure.csv
// -------------------------------

var sites = aoiSitesBuffered();

Map.centerObject(sites, 3);
Map.addLayer(sites, {}, 'Oil infrastructure AOIs');
print('AOI source:', USE_EE_ASSET ? AOI_ASSET_ID : 'embedded CSV sync');


// -------------------------------
// 3. Sentinel-2 collections
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
  primary: s2Sr,
  secondary: s2Clouds,
  condition: ee.Filter.equals({
    leftField: 'system:index',
    rightField: 'system:index'
  })
}));


// -------------------------------
// 4. Cloud mask and spectral indices
// -------------------------------

function addCloudProbability(img) {
  var cloudProbImg = ee.Image(img.get('s2cloudless'));
  var cloudProb = cloudProbImg.select('probability').rename('cloud_probability');
  return img.addBands(cloudProb);
}

function maskClouds(img) {
  var cloudProb = img.select('cloud_probability');
  var isNotCloud = cloudProb.lt(CLD_PRB_THRESH);

  var scl = img.select('SCL');
  var sclMask = scl.neq(3)
    .and(scl.neq(8))
    .and(scl.neq(9))
    .and(scl.neq(10))
    .and(scl.neq(11));

  return img.updateMask(isNotCloud).updateMask(sclMask);
}

function addIndices(img) {
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI');

  var bsi = img.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    {
      SWIR: img.select('B11'),
      RED: img.select('B4'),
      NIR: img.select('B8'),
      BLUE: img.select('B2')
    }
  ).rename('BSI');

  return img
    .addBands([ndvi, ndwi, ndbi, bsi])
    .copyProperties(img, ['system:time_start', 'system:index']);
}

var s2Processed = joined
  .map(addCloudProbability)
  .map(maskClouds)
  .map(addIndices)
  .select(['NDVI', 'NDWI', 'NDBI', 'BSI', 'cloud_probability']);


// -------------------------------
// 5. Create monthly composites with mean + stdDev
// -------------------------------

var start = ee.Date(START_DATE);
var end = ee.Date(END_DATE);
var nMonths = end.difference(start, 'month').round();
var monthList = ee.List.sequence(0, nMonths.subtract(1));

var monthlyFeatures = ee.FeatureCollection(
  monthList.map(function(m) {
    var startMonth = start.advance(m, 'month');
    var endMonth = startMonth.advance(1, 'month');

    var monthlyCol = s2Processed.filterDate(startMonth, endMonth);
    var validObsCount = monthlyCol.size();

    var monthlyMedian = monthlyCol.median();
    var monthlyStdDev = monthlyCol.reduce(ee.Reducer.stdDev());

    var combinedReducer = ee.Reducer.mean().combine({
      reducer2: ee.Reducer.stdDev(),
      sharedInputs: true
    });

    var stats = monthlyMedian.reduceRegions({
      collection: sites,
      reducer: ee.Reducer.mean(),
      scale: 20,
      tileScale: 4
    });

    var statsStd = monthlyStdDev.reduceRegions({
      collection: sites,
      reducer: ee.Reducer.mean(),
      scale: 20,
      tileScale: 4
    });

    stats = stats.map(function(f) {
      var siteId = f.get('site_id');
      var matchStd = statsStd.filter(ee.Filter.eq('site_id', siteId)).first();
      return f.set({
        year: startMonth.get('year'),
        month: startMonth.get('month'),
        date_month: startMonth.format('YYYY-MM'),
        valid_obs_count: validObsCount,
        sensor: 'Sentinel-2',
        NDVI_std: matchStd.get('NDVI_stdDev'),
        NDWI_std: matchStd.get('NDWI_stdDev'),
        NDBI_std: matchStd.get('NDBI_stdDev'),
        BSI_std: matchStd.get('BSI_stdDev')
      });
    });

    return stats;
  })
).flatten();


// -------------------------------
// 6. Inspect output
// -------------------------------

print('Monthly Sentinel-2 features (11 AOIs)', monthlyFeatures.limit(20));

Map.addLayer(
  s2Processed.filterDate('2024-01-01', '2024-12-31').median(),
  {bands: ['NDVI'], min: -0.5, max: 0.8},
  'Example NDVI 2024'
);


// -------------------------------
// 7. Export to Google Drive
// -------------------------------

Export.table.toDrive({
  collection: monthlyFeatures,
  description: 'sentinel2_oil_sites_monthly_indices_201704_202512_11aoi',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'sentinel2_oil_sites_monthly_indices_201704_202512_11aoi',
  fileFormat: 'CSV'
});
