package com.example.agrimitra;



import android.content.Intent;
import android.os.Bundle;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.SeekBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class SimulatorActivity extends AppCompatActivity {

    private static final int SPLASH_TIME = 5000;
    private TranslationHelper translationHelper;

    EditText landInput;
    SeekBar waterSeekBar, nSeekBar, pSeekBar, kSeekBar;
    TextView waterLabel, nLabel, pLabel, kLabel, tvResult;
    Spinner seasonSpinner, soilSpinner;
    CardView simulateButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_simulator);

        translationHelper = new TranslationHelper(this);

        // Initialize views
        landInput = findViewById(R.id.landInput);
        waterSeekBar = findViewById(R.id.waterSeekBar);
        nSeekBar = findViewById(R.id.nSeekBar);
        pSeekBar = findViewById(R.id.pSeekBar);
        kSeekBar = findViewById(R.id.kSeekBar);

        waterLabel = findViewById(R.id.waterLabel);
        nLabel = findViewById(R.id.nLabel);
        pLabel = findViewById(R.id.pLabel);
        kLabel = findViewById(R.id.kLabel);

        seasonSpinner = findViewById(R.id.seasonSpinner);
        soilSpinner = findViewById(R.id.soilSpinner);
        simulateButton = findViewById(R.id.simulateButton);

        // Setup Spinner Values
        String[] seasons = {"Summer", "Winter", "Spring", "Rainy"};
        String[] soils = {"Sandy", "Clay", "Loamy"};

        translationHelper.initializeTranslation(() -> {
            ArrayAdapter<String> seasonAdapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_item,   // custom layout for text
                    seasons
            );
            seasonAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);

            ArrayAdapter<String> soilAdapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_item,   // custom layout for text
                    soils
            );
            soilAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);

            seasonSpinner.setAdapter(seasonAdapter);
            soilSpinner.setAdapter(soilAdapter);
        });

        // Water SeekBar
        waterSeekBar.setMax(5000);
        waterSeekBar.setProgress(1000);
        waterLabel.setText("Water (Kilolitres/acre): " + waterSeekBar.getProgress());
        waterSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                waterLabel.setText("Water (Kilolitres/acre): " + progress);
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // N, P, K SeekBars
        nSeekBar.setMax(150); nSeekBar.setProgress(80);
        nSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                nLabel.setText("Nitrogen Fertilizer (N): " + progress + " kg/acre");
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        pSeekBar.setMax(90); pSeekBar.setProgress(40);
        pSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                pLabel.setText("Phosphorus Fertilizer(P): " + progress + " kg/acre");
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        kSeekBar.setMax(150); kSeekBar.setProgress(80);
        kSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                kLabel.setText("Potassium Fertilizer(K): " + progress + " kg/acre");
            }
            @Override public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override public void onStopTrackingTouch(SeekBar seekBar) {}
        });

        // Button Click
        simulateButton.setOnClickListener(v -> {
            String landStr = landInput.getText().toString();
            if (landStr.isEmpty()) {
                translationHelper.showTranslatedToast("Please enter land area");
                return;
            }

            double land = Double.parseDouble(landStr);
            int water = waterSeekBar.getProgress();
            water=water*1000;
            double N = nSeekBar.getProgress();
            double P = pSeekBar.getProgress();
            double K = kSeekBar.getProgress();
            String season = seasonSpinner.getSelectedItem().toString();
            String soil = soilSpinner.getSelectedItem().toString();

            // Water Requirement
            double waterReqMM;
            switch (season) {
                case "Winter":
                    waterReqMM = 500;
                    break;
                case "Spring":
                    waterReqMM = 600;
                    break;
                case "Rainy":
                    waterReqMM = 550;
                    break;
                case "Summer":
                    waterReqMM = 700;
                    break;
                default:
                    waterReqMM = 600;
                    break;
            }

// Soil adjustment
            if (soil.equals("Sandy")) {
                waterReqMM *= 1.2;
            } else if (soil.equals("Clay")) {
                waterReqMM *= 0.9;
            }

// Convert mm → liters/acre
            double waterReqLiters = waterReqMM * 4046.86;
            double f_water = Math.min(1.0, water / waterReqLiters);

// Soil factor
            double f_soil;
            switch (soil) {
                case "Loamy":
                    f_soil = 1.2;
                    break;
                case "Clay":
                    f_soil = 0.9;
                    break;
                case "Sandy":
                    f_soil = 0.8;
                    break;
                default:
                    f_soil = 1.0;
                    break;
            }

// Season factor
            double f_season;
            switch (season) {
                case "Winter":
                    f_season = 1.1;
                    break;
                case "Summer":
                    f_season = 0.9;
                    break;
                case "Spring":
                    f_season = 1.0;
                    break;
                case "Rainy":
                    f_season = 1.05;
                    break;
                default:
                    f_season = 1.0;
                    break;
            }


            // Fertilizer factor (quadratic response)

            // ---------- Step 1: Baseline Optimums ----------
            double N_opt = 80, P_opt = 40, K_opt = 80;

            // ---------- Step 2: Soil Adjustments ----------
            switch (soil) {
                case "Sandy":
                    N_opt *= 1.2; K_opt *= 1.2; break;
                case "Clay":
                    N_opt *= 0.9; P_opt *= 0.9; break;
                default: break; // Loamy or others → no change
            }

            // ---------- Step 3: Climate Adjustments ----------
            switch (season) {
                case "Rainy":
                    N_opt *= 1.1; K_opt *= 1.1; break;
                case "Summer":
                    K_opt *= 1.2; break;
                default: break;
            }


            // ---------- Step 5: Quadratic Response Function ----------
            double f_N = (N / N_opt) * (2 - (N / N_opt));
            double f_P = (P / P_opt) * (2 - (P / P_opt));
            double f_K = (K / K_opt) * (2 - (K / K_opt));

            // Clamp values between 0 and 1
            f_N = Math.max(0.0, Math.min(1.0, f_N));
            f_P = Math.max(0.0, Math.min(1.0, f_P));
            f_K = Math.max(0.0, Math.min(1.0, f_K));

            // ---------- Step 6: Combine Nutrient Factors ----------
            double f_nutrient;

            if (N == 0 && P == 0 && K == 0) {
                f_nutrient = 0;
            } else if (P == 0 && K == 0) {
                f_nutrient = f_N;
            } else if (N == 0 && K == 0) {
                f_nutrient = f_P;
            } else if (N == 0 && P == 0) {
                f_nutrient = f_K;
            } else if (K == 0) {
                f_nutrient = Math.sqrt(f_N * f_P);
            } else if (P == 0) {
                f_nutrient = Math.sqrt(f_N * f_K);
            } else if (N == 0) {
                f_nutrient = Math.sqrt(f_P * f_K);
            } else {
                f_nutrient = Math.cbrt(f_N * f_P * f_K);
            }




            // Potential yield
            double Y_pot = 10.0; // tons/acre

            // Predicted yield
            double predictedYield = Y_pot * land * f_water * f_nutrient * f_soil * f_season;


            Intent intent=new Intent(SimulatorActivity.this, LoadingActivity.class);
            intent.putExtra("yieldperacre",predictedYield/land);
            intent.putExtra("totalyield",predictedYield);
            startActivity(intent);

            // Display
//            tvResult.setText("Predicted Total Yield: " + String.format("%.2f", predictedYield) + " tons\nPredicted Yield per Acre: "+String.format("%.2f", predictedYield/land)+" tons/acre");
//            tvResult.setVisibility(TextView.VISIBLE);
        });

        // Edge padding for system bars
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
