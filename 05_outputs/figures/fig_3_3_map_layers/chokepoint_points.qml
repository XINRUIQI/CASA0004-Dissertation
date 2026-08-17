<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 type="singleSymbol" symbollevels="0" forceraster="0" enableorderby="0">
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleMarker" locked="0" enabled="1" pass="0">
          <prop k="name" v="diamond"/>
          <prop k="color" v="209,98,43,255"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.4"/>
          <prop k="size" v="3.6"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fontFamily="Arial" fontSize="8" fontWeight="75" textColor="138,61,22,255" namedStyle="Bold"/>
      <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,255"/>
      <placement placement="2" dist="1.6" distUnits="MM"/>
      <rendering drawLabels="1"/>
    </settings>
  </labeling>
  <customproperties>
    <Option type="Map">
      <Option name="labeling/fieldName" type="QString" value="map_label"/>
    </Option>
  </customproperties>
</qgis>
