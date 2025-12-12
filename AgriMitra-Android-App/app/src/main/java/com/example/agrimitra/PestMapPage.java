package com.example.agrimitra;


import android.graphics.Color;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.fragment.app.FragmentActivity;

import com.google.android.gms.maps.CameraUpdateFactory;
import com.google.android.gms.maps.GoogleMap;
import com.google.android.gms.maps.OnMapReadyCallback;
import com.google.android.gms.maps.SupportMapFragment;
import com.google.android.gms.maps.model.LatLng;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.maps.android.heatmaps.Gradient;
import com.google.maps.android.heatmaps.HeatmapTileProvider;
import com.google.maps.android.heatmaps.WeightedLatLng;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PestMapPage extends FragmentActivity implements OnMapReadyCallback {

    private GoogleMap mMap;
    private HeatmapTileProvider mProvider;

    // Map to store district report counts
    private final Map<String, Integer> districtReports = new HashMap<>();

    // Map of district coordinates
    private final Map<String, LatLng> districtCoords = new HashMap<String, LatLng>() {{
        put("chennai", new LatLng(13.0827, 80.2707));
        put("coimbatore", new LatLng(11.0168, 76.9558));
        put("madurai", new LatLng(9.9252, 78.1198));
        put("trichy", new LatLng(10.7905, 78.7047));
        put("salem", new LatLng(11.6643, 78.1460));
        put("erode", new LatLng(11.3385, 77.7172));
        put("tirunelveli", new LatLng(8.7285, 77.7081));
        put("vellore", new LatLng(12.9165, 79.1325));
        put("thanjavur", new LatLng(10.7867, 79.1378));
    }};

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_pest_map_page);

        SupportMapFragment mapFragment = (SupportMapFragment) getSupportFragmentManager()
                .findFragmentById(R.id.mapFragment);
        mapFragment.getMapAsync(this);

        // Initialize all district reports with 0
        for (String district : districtCoords.keySet()) {
            districtReports.put(district, 0);
        }

        loadReportsFromFirebase();
    }

    private void loadReportsFromFirebase() {
        DatabaseReference reportsRef = FirebaseDatabase.getInstance().getReference("Pest-report");

        reportsRef.addListenerForSingleValueEvent(new ValueEventListener() {
            @Override
            public void onDataChange(@NonNull DataSnapshot snapshot) {
                for (String district : districtCoords.keySet()) {
                    if (snapshot.child(district).exists()) {
                        Integer value = snapshot.child(district).getValue(Integer.class);
                        if (value != null) {
                            districtReports.put(district, value);
                        }
                    }
                }

                // After data is loaded, add heatmap if map is ready
                if (mMap != null) {
                    addHeatmap();
                }
            }

            @Override
            public void onCancelled(@NonNull DatabaseError error) {
                // Handle errors
            }
        });
    }

    @Override
    public void onMapReady(GoogleMap googleMap) {
        mMap = googleMap;

        // Move camera to Tamil Nadu
        LatLng tamilNadu = new LatLng(11.1271, 78.6569);
        mMap.moveCamera(CameraUpdateFactory.newLatLngZoom(tamilNadu, 6.5f));

        // If Firebase data already loaded, show heatmap
        if (!districtReports.isEmpty() && districtReports.values().stream().anyMatch(v -> v > 0)) {
            addHeatmap();
        }
    }

    private void addHeatmap() {
        List<WeightedLatLng> heatMapData = new ArrayList<>();

        for (Map.Entry<String, LatLng> entry : districtCoords.entrySet()) {
            String district = entry.getKey();
            LatLng coord = entry.getValue();
            int weight = districtReports.getOrDefault(district, 0);

            if (weight > 0) {
                heatMapData.add(new WeightedLatLng(coord, weight));
            }
        }

        if (!heatMapData.isEmpty()) {
            int[] colors = {Color.GREEN, Color.RED};
            float[] startPoints = {0.2f, 1f};
            Gradient gradient = new Gradient(colors, startPoints);

            mProvider = new HeatmapTileProvider.Builder()
                    .weightedData(heatMapData)
                    .gradient(gradient)
                    .radius(50)
                    .opacity(0.7)
                    .build();

            mMap.addTileOverlay(new com.google.android.gms.maps.model.TileOverlayOptions()
                    .tileProvider(mProvider));
        }
    }
}
