package com.example.agrimitra;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import android.app.ProgressDialog;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

public class CropRecommendation extends AppCompatActivity {

    private TranslationHelper translationHelper;

    private EditText nitrogenInput, phosphorusInput, potassiumInput, tempInput, humidityInput, phInput, rainfallInput;
    private TextView nitrogenLabel, phosphorusLabel, potassiumLabel, tempLabel, humidityLabel, phLabel, rainfallLabel, submitText;
    private CardView submitBtn;
    private ProgressDialog progressDialog;

    private static final String PREDICT_URL = "https://crop-detection-model.onrender.com/predict";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_crop_recommendation);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);



        // Initialize inputs
        nitrogenInput = findViewById(R.id.nitrogenInput);
        phosphorusInput = findViewById(R.id.phosphorusInput);
        potassiumInput = findViewById(R.id.potassiumInput);
        tempInput = findViewById(R.id.tempInput);
        humidityInput = findViewById(R.id.humidityInput);
        phInput = findViewById(R.id.phInput);
        rainfallInput = findViewById(R.id.rainfallInput);
        submitBtn = findViewById(R.id.submitBtn);
        submitText = findViewById(R.id.submit_tt);

        // Initialize labels
        nitrogenLabel = findViewById(R.id.nitrogenLabel);
        phosphorusLabel = findViewById(R.id.phosphorusLabel);
        potassiumLabel = findViewById(R.id.potassiumLabel);
        tempLabel = findViewById(R.id.tempLabel);
        humidityLabel = findViewById(R.id.humidityLabel);
        phLabel = findViewById(R.id.phLabel);
        rainfallLabel = findViewById(R.id.rainfallLabel);

        CardView fetch = findViewById(R.id.btnFetchData);

        fetch.setOnClickListener(v -> {
            ProgressDialog pd = new ProgressDialog(CropRecommendation.this);
            translationHelper.translateText("Fetching data...", translatedText -> pd.setMessage(translatedText));
            pd.setCancelable(false);
            pd.show();

            new Handler().postDelayed(() -> {
                pd.dismiss();
                // Fill inputs with dummy data
                nitrogenInput.setText("78.8");
                phosphorusInput.setText("63.2");
                potassiumInput.setText("54.8");
                tempInput.setText("27.0");
                humidityInput.setText("78.6");
                rainfallInput.setText("1023.6");
                phInput.setText("6.17");
            }, 2000);
        });

        progressDialog = new ProgressDialog(this);
        progressDialog.setCancelable(false);

        // Translate UI elements
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(nitrogenLabel, "Nitrogen (N)");
            translationHelper.translateTextView(phosphorusLabel, "Phosphorus (P)");
            translationHelper.translateTextView(potassiumLabel, "Potassium (K)");
            translationHelper.translateTextView(tempLabel, "Temperature (°C)");
            translationHelper.translateTextView(humidityLabel, "Moisture (%)");
            translationHelper.translateTextView(phLabel, "pH Level");
            translationHelper.translateTextView(rainfallLabel, "Rainfall (mm)");
            translationHelper.translateTextView(submitText, "Submit");

            // Translate hints
            nitrogenInput.setHint("Enter Nitrogen value");
            phosphorusInput.setHint("Enter Phosphorus value");
            potassiumInput.setHint("Enter Potassium value");
            tempInput.setHint("Enter Temperature in °C");
            humidityInput.setHint("Enter Humidity %");
            phInput.setHint("Enter pH Level");
            rainfallInput.setHint("Enter Rainfall in mm");

            translationHelper.translateText("Predicting crop...", translatedText -> progressDialog.setMessage(translatedText));
        });

        submitBtn.setOnClickListener(v -> sendPredictionRequest());

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void sendPredictionRequest() {
        String N = nitrogenInput.getText().toString().trim();
        String P = phosphorusInput.getText().toString().trim();
        String K = potassiumInput.getText().toString().trim();
        String temperature = tempInput.getText().toString().trim();
        String humidity = humidityInput.getText().toString().trim();
        String ph = phInput.getText().toString().trim();
        String rainfall = rainfallInput.getText().toString().trim();

        if (N.isEmpty() || P.isEmpty() || K.isEmpty() || temperature.isEmpty() ||
                humidity.isEmpty() || ph.isEmpty() || rainfall.isEmpty()) {
            translationHelper.showTranslatedToast("Please enter all values");
            return;
        }

        progressDialog.show();

        try {
            JSONObject jsonRequest = new JSONObject();
            // Parse as double to support decimal values
            jsonRequest.put("N", Double.parseDouble(N));
            jsonRequest.put("P", Double.parseDouble(P));
            jsonRequest.put("K", Double.parseDouble(K));
            jsonRequest.put("temperature", Double.parseDouble(temperature));
            jsonRequest.put("humidity", Double.parseDouble(humidity));
            jsonRequest.put("ph", Double.parseDouble(ph));
            jsonRequest.put("rainfall", Double.parseDouble(rainfall));

            JsonObjectRequest request = new JsonObjectRequest(Request.Method.POST, PREDICT_URL, jsonRequest,
                    response -> {
                        progressDialog.dismiss();
                        try {
                            String crop = response.getString("predicted_crop");
                            Intent intent = new Intent(CropRecommendation.this, CropRecommendationResultPage.class);
                            intent.putExtra("crop", crop);
                            startActivity(intent);

                            translationHelper.showTranslatedToast("Predicted Crop: " + crop, Toast.LENGTH_LONG);
                        } catch (JSONException e) {
                            translationHelper.showTranslatedToast("Response parsing error");
                        }
                    },
                    error -> {
                        progressDialog.dismiss();
                        translationHelper.showTranslatedToast("Error: " + error.getMessage(), Toast.LENGTH_LONG);
                    });

            RequestQueue queue = Volley.newRequestQueue(this);
            queue.add(request);

        } catch (JSONException e) {
            progressDialog.dismiss();
            e.printStackTrace();
            Log.d("error", e.getMessage());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
