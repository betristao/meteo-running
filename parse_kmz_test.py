import xml.etree.ElementTree as ET
import zipfile
import io

def parse_kml_string(kml_string):
    points = []
    try:
        root = ET.fromstring(kml_string)
        # KML usually has coordinates inside <LineString><coordinates>
        # The format is "lon,lat,alt lon,lat,alt ..."
        for coords in root.iter():
            if 'coordinates' in coords.tag:
                text = coords.text.strip()
                pairs = text.split()
                for pair in pairs:
                    parts = pair.split(',')
                    if len(parts) >= 2:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        ele = float(parts[2]) if len(parts) >= 3 else 0.0
                        points.append((lat, lon, ele))
                if len(points) > 0:
                    break # Usually just need the first major track
    except Exception as e:
        print("XML Error:", e)
    return points

def extract_kmz(kmz_bytes):
    try:
        with zipfile.ZipFile(io.BytesIO(kmz_bytes)) as z:
            for filename in z.namelist():
                if filename.lower().endswith('.kml'):
                    kml_data = z.read(filename)
                    return parse_kml_string(kml_data)
    except Exception as e:
        print("Zip Error:", e)
    return []

# Test with mock kml data
kml_mock = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <LineString>
        <coordinates>
          -9.142685,38.736946,0 -9.142,38.737,0 -9.14,38.74,0
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>
"""
print("KML Parse:", parse_kml_string(kml_mock))
