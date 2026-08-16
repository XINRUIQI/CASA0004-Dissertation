<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.0" styleCategories="Symbology|Labeling" labelsEnabled="1">
  <renderer-v2 type="categorizedSymbol" attr="site_type" symbollevels="0" forceraster="0" enableorderby="0">
    <categories>
      <category value="port" label="AOI — port" render="true" symbol="0"/>
      <category value="terminal" label="AOI — terminal" render="true" symbol="1"/>
      <category value="refinery" label="AOI — refinery" render="true" symbol="2"/>
    </categories>
    <symbols>
      <symbol type="marker" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleMarker" locked="0" enabled="1" pass="0">
          <prop k="name" v="circle"/>
          <prop k="color" v="46,90,136,255"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.4"/>
          <prop k="size" v="3.2"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="1" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleMarker" locked="0" enabled="1" pass="0">
          <prop k="name" v="square"/>
          <prop k="color" v="46,90,136,255"/>
          <prop k="outline_color" v="255,255,255,255"/>
          <prop k="outline_width" v="0.4"/>
          <prop k="size" v="3.0"/>
          <prop k="size_unit" v="MM"/>
        </layer>
      </symbol>
      <symbol type="marker" name="2" alpha="1" clip_to_extent="1" force_rhr="0">
        <layer class="SimpleMarker" locked="0" enabled="1" pass="0">
          <prop k="name" v="triangle"/>
          <prop k="color" v="46,90,136,255"/>
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
      <text-style fontFamily="Arial" fontSize="8" fontWeight="50" textColor="31,62,95,255" namedStyle="Regular"/>
      <text-buffer bufferDraw="1" bufferSize="0.8" bufferColor="255,255,255,255"/>
      <placement placement="2" dist="1.4" distUnits="MM" predefinedPositionOrder="TR,TL,BR,BL,R,L,T,B"/>
      <rendering drawLabels="1" minFeatureSize="0" fontLimitPixelSize="0"/>
    </settings>
  </labeling>
  <customproperties>
    <Option type="Map">
      <Option name="labeling/fieldName" type="QString" value="map_label"/>
    </Option>
  </customproperties>
</qgis>
