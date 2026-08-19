// =======================================================
// VIIRS Monthly Night-time Lights — Oil Infrastructure AOIs
// Dataset: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
// Period: 2012-04 ~ 2025-12
// Output: monthly mean/max radiance per AOI
// =======================================================


// -------------------------------
// 1. User settings
// -------------------------------

var START_DATE = '2012-04-01';
var END_DATE   = '2025-12-31';

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


// -------------------------------
// 3. Load VIIRS monthly composites
// avg_rad = monthly average radiance (nW/cm²/sr)
// cf_cvg  = cloud-free observation count
// -------------------------------

var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate(START_DATE, END_DATE)
  .select(['avg_rad', 'cf_cvg']);

print('Total VIIRS monthly images:', viirs.size());


// -------------------------------
// 4. Quality mask: remove negative radiance
//    (stray light, data artefacts)
// -------------------------------

function maskNegativeRadiance(img) {
  var mask = img.select('avg_rad').gte(0);
  return img.updateMask(mask);
}

var viirsMasked = viirs.map(maskNegativeRadiance);


// -------------------------------
// 5. Extract zonal statistics per AOI per month
// Uses mean + max radiance and mean cloud-free count
// -------------------------------

var monthlyFeatures = viirsMasked.map(function(img) {
  var date = img.date();

  var combinedReducer = ee.Reducer.mean()
    .combine({reducer2: ee.Reducer.max(), sharedInputs: true})
    .combine({reducer2: ee.Reducer.stdDev(), sharedInputs: true});

  var stats = img.reduceRegions({
    collection: sites,
    reducer: combinedReducer,
    scale: 500,
    tileScale: 2
  });

  stats = stats.map(function(f) {
    return f.set({
      year: date.get('year'),
      month: date.get('month'),
      date_month: date.format('YYYY-MM'),
      sensor: 'VIIRS'
    });
  });

  return stats;
}).flatten();


// -------------------------------
// 6. Rename output bands for clarity
// -------------------------------

monthlyFeatures = monthlyFeatures.map(function(f) {
  return ee.Feature(null, {
    site_id:        f.get('site_id'),
    site_name:      f.get('site_name'),
    site_type:      f.get('site_type'),
    country:        f.get('country'),
    region:         f.get('region'),
    year:           f.get('year'),
    month:          f.get('month'),
    date_month:     f.get('date_month'),
    sensor:         f.get('sensor'),
    ntl_avg_rad_mean:   f.get('avg_rad_mean'),
    ntl_avg_rad_max:    f.get('avg_rad_max'),
    ntl_avg_rad_stddev: f.get('avg_rad_stdDev'),
    ntl_cf_cvg_mean:    f.get('cf_cvg_mean')
  });
});


// -------------------------------
// 7. Inspect and export
// -------------------------------

print('Monthly VIIRS nightlight features (11 AOIs)', monthlyFeatures.limit(20));

Export.table.toDrive({
  collection: monthlyFeatures,
  description: 'viirs_oil_sites_monthly_nightlights_201401_202512_11aoi',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'viirs_oil_sites_monthly_nightlights_201401_202512_11aoi',
  fileFormat: 'CSV'
});
