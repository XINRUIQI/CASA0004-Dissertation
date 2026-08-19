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


// --- script body (from extract_viirs_monthly_nightlights_gee.js) ---

// =======================================================
// VIIRS Monthly Night-time Lights — Oil Infrastructure AOIs
// Dataset: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG
// Period: 2012-04 ~ 2025-12
// Output: monthly mean/max radiance per AOI
// AOIs: aoi_oil_infrastructure.csv
//
// PREREQUISITE: load_aoi_config_gee.js in the same GEE project (or paste above).
// Regenerate from CSV: python sync_aoi_csv_to_gee.py
// =======================================================


// -------------------------------
// 1. User settings
// -------------------------------

var START_DATE = '2012-04-01';
var END_DATE   = '2025-12-31';

var EXPORT_FOLDER = 'CASA0004_oil_project';


// -------------------------------
// 2. AOI table — from aoi_oil_infrastructure.csv
// -------------------------------

var sites = aoiSitesBuffered();

Map.centerObject(sites, 3);
print('AOI source:', USE_EE_ASSET ? AOI_ASSET_ID : 'embedded CSV sync');


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
