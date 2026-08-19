// =======================================================
// Sentinel-2 Monthly Oil Infrastructure Indicators
// Output: monthly NDVI, NDWI, NDBI, BSI by oil-related AOI
// Dataset: COPERNICUS/S2_SR_HARMONIZED + S2 Cloud Probability
// Period: 2017-04 ~ 2025-12
// AOIs: 11 global oil infrastructure sites
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
// 2. AOI table — from aoi_oil_infrastructure.csv (uploaded as a GEE asset)
// -------------------------------
// GEE cannot read a local CSV, so upload aoi_oil_infrastructure.csv as a table
// asset (ingest lon -> longitude, lat -> latitude) and set AOI_ASSET to its ID.
// The 5 km circular buffer comes from the CSV's buffer_km column.

var AOI_ASSET = 'projects/ee-USERNAME/assets/aoi_oil_infrastructure';

var sites = ee.FeatureCollection(AOI_ASSET)
  .select(['site_id', 'site_name', 'site_type', 'country', 'region', 'buffer_km'])
  .map(function(f) {
    return f.buffer(ee.Number(f.get('buffer_km')).multiply(1000));
  });

Map.centerObject(sites, 3);
Map.addLayer(sites, {}, 'Oil infrastructure AOIs');


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
