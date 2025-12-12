package com.example.agrimitra;

import android.app.ProgressDialog;
import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;

import org.json.JSONObject;

import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class YieldPrediction extends AppCompatActivity {

    private TranslationHelper translationHelper;
    private Intent mainIntent;

    Spinner spinnerCropType, spinnerSoilType;
    EditText soilPhInput, soilQualityInput, soilTempInput, soilHumidityInput,
            windSpeedInput, nitrogenInput, phosphorusInput, potassiumInput, rainfallInput;
    TextView cropTypeLabel, soilTypeLabel, soilPhLabel, soilQualityLabel, soilTempLabel,
            soilHumidityLabel, windSpeedLabel, nitrogenLabel, phosphorusLabel, potassiumLabel, rainfallLabel;
    TextView fetchText, submitText;

    CardView fetch, submit;

    private final OkHttpClient client = new OkHttpClient();
    private static final String BASE_URL = "https://yield-prediction-1-9a30.onrender.com/predict";
    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    // Flags to control single launch
    private boolean predictionReady = false;

    private DatabaseReference npkRef;
    private boolean agentReady = false;

    double humidity=0,moisture=0,nitrogen=0,pH=0,phosphorus=0,potassium=0,temperature=0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_yield_prediction);

        mainIntent = new Intent(YieldPrediction.this, YieldPredictionResult.class);

        translationHelper = new TranslationHelper(this);

        fetch = findViewById(R.id.btnFetchData);
        submit = findViewById(R.id.btnSubmit);

        spinnerCropType = findViewById(R.id.spinnerCropType);
        spinnerSoilType = findViewById(R.id.spinnerSoilType);

        cropTypeLabel = findViewById(R.id.cropTypeLabel);
        soilTypeLabel = findViewById(R.id.soilTypeLabel);
        soilPhLabel = findViewById(R.id.soilPhLabel);
        soilQualityLabel = findViewById(R.id.soilQualityLabel);
        soilTempLabel = findViewById(R.id.soilTempLabel);
        soilHumidityLabel = findViewById(R.id.soilHumidityLabel);
        windSpeedLabel = findViewById(R.id.windSpeedLabel);
        nitrogenLabel = findViewById(R.id.nitrogenLabel);
        phosphorusLabel = findViewById(R.id.phosphorusLabel);
        potassiumLabel = findViewById(R.id.potassiumLabel);
        rainfallLabel = findViewById(R.id.rainfallLabel);
        fetchText = fetch.findViewById(R.id.text1);
        submitText = submit.findViewById(R.id.text2);


        FirebaseDatabase database = FirebaseDatabase.getInstance();
        npkRef = database.getReference("NPK");

        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(cropTypeLabel, "Crop Type");
            translationHelper.translateTextView(soilTypeLabel, "Soil Type");
            translationHelper.translateTextView(soilPhLabel, "Soil pH (0-14)");
            translationHelper.translateTextView(soilQualityLabel, "Soil Quality (0-100)");
            translationHelper.translateTextView(soilTempLabel, "Soil Temperature (°C)");
            translationHelper.translateTextView(soilHumidityLabel, "Soil Humidity (0-100%)");
            translationHelper.translateTextView(windSpeedLabel, "Wind Speed (km/h)");
            translationHelper.translateTextView(nitrogenLabel, "Nitrogen (0-200)");
            translationHelper.translateTextView(phosphorusLabel, "Phosphorus (0-200)");
            translationHelper.translateTextView(potassiumLabel, "Potassium (0-200)");
            translationHelper.translateTextView(rainfallLabel, "Average Rainfall (0-1000 mm)");
            translationHelper.translateTextView(fetchText, "Fetch Data");
            translationHelper.translateTextView(submitText, "Submit");

            String[] cropTypes = {"Select crop type", "Barley", "Corn", "Cotton", "Potato", "Rice", "Soybean", "Sugarcane", "Sunflower", "Tomato", "Wheat"};
            ArrayAdapter<String> cropAdapter = new ArrayAdapter<>(this, R.layout.spinner_item, cropTypes);
            cropAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerCropType.setAdapter(cropAdapter);

            String[] soilTypes = {"Select soil type", "Clay", "Loamy", "Peaty", "Sandy", "Saline"};
            ArrayAdapter<String> soilAdapter = new ArrayAdapter<>(this, R.layout.spinner_item, soilTypes);
            soilAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerSoilType.setAdapter(soilAdapter);
        });

        soilPhInput = findViewById(R.id.soilPhInput);
        soilQualityInput = findViewById(R.id.soilQualityInput);
        soilTempInput = findViewById(R.id.soilTempInput);
        soilHumidityInput = findViewById(R.id.soilHumidityInput);
        windSpeedInput = findViewById(R.id.windSpeedInput);
        nitrogenInput = findViewById(R.id.nitrogenInput);
        phosphorusInput = findViewById(R.id.phosphorusInput);
        potassiumInput = findViewById(R.id.potassiumInput);
        rainfallInput = findViewById(R.id.rainfallInput);

        fetch.setOnClickListener(v -> {
            ProgressDialog progressDialog = new ProgressDialog(YieldPrediction.this);
            translationHelper.translateText("Fetching data...", translatedText -> progressDialog.setMessage(translatedText));
            progressDialog.setCancelable(false);
            progressDialog.show()
            ;


            npkRef.addListenerForSingleValueEvent(new ValueEventListener() {
                @Override
                public void onDataChange(DataSnapshot snapshot) {
                    if (snapshot.exists()) {
                        humidity = snapshot.child("humidity").getValue(Double.class);
                        moisture = snapshot.child("moisture").getValue(Double.class);
                        nitrogen = snapshot.child("nitrogen").getValue(Double.class);
                        pH = snapshot.child("pH").getValue(Double.class);
                        phosphorus = snapshot.child("phosphorus").getValue(Double.class);
                        potassium = snapshot.child("potassium").getValue(Double.class);
                        temperature = snapshot.child("temperature").getValue(Double.class);


                        soilPhInput.setText(String.valueOf(pH));

                        soilQualityInput.setText(String.valueOf("67.2"));
                        soilTempInput.setText(String.valueOf(temperature));
                        soilHumidityInput.setText(String.valueOf(humidity));
                        windSpeedInput.setText(String.valueOf(10.2));
                        nitrogenInput.setText(String.valueOf(nitrogen));
                        phosphorusInput.setText(String.valueOf(phosphorus));
                        potassiumInput.setText(String.valueOf(potassium));
                        rainfallInput.setText(String.valueOf("187.0"));

                        progressDialog.dismiss();



                        // Use these values as needed
                        Log.d("FirebaseData", "Humidity: " + humidity + ", pH: " + pH);
                    } else {
                        Log.d("FirebaseData", "No data found under NPK");
                    }
                }

                @Override
                public void onCancelled(DatabaseError error) {
                    Log.e("FirebaseData", "Failed to read data: " + error.getMessage());
                }
            });





        });

        submit.setOnClickListener(v -> {
            predictionReady = false;
            agentReady = false;

            sendPredictionRequest();
            sendAgentRequest();
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void checkAndLaunch() {
        if (predictionReady && agentReady) {
            startActivity(mainIntent);
        }
    }

    private void sendPredictionRequest() {
        try {
            String cropType = spinnerCropType.getSelectedItem().toString();
            String soilType = spinnerSoilType.getSelectedItem().toString();

            if (cropType.equals("Select crop type") || soilType.equals("Select soil type")) {
                translationHelper.showTranslatedToast("Please select valid Crop and Soil type");
                return;
            }

            JSONObject json = new JSONObject();
            json.put("crop_type", cropType);
            json.put("soil_type", soilType);
            json.put("soil_ph", Double.parseDouble(soilPhInput.getText().toString()));
            json.put("temperature", Double.parseDouble(soilTempInput.getText().toString()));
            json.put("humidity", Double.parseDouble(soilHumidityInput.getText().toString()));
            json.put("wind_speed", Double.parseDouble(windSpeedInput.getText().toString()));
            json.put("n", Double.parseDouble(nitrogenInput.getText().toString()));
            json.put("p", Double.parseDouble(phosphorusInput.getText().toString()));
            json.put("k", Double.parseDouble(potassiumInput.getText().toString()));
            json.put("soil_quality", Double.parseDouble(soilQualityInput.getText().toString()));
            json.put("avg_rainfall", Double.parseDouble(rainfallInput.getText().toString()));

            RequestBody body = RequestBody.create(json.toString(), JSON);
            Request request = new Request.Builder().url(BASE_URL).post(body).build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    runOnUiThread(() -> translationHelper.showTranslatedToast("API Error: " + e.getMessage()));
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (!response.isSuccessful()) {
                        runOnUiThread(() -> translationHelper.showTranslatedToast("Server Error: " + response.code()));
                    } else {
                        try {
                            String resBody = response.body().string();
                            JSONObject resJson = new JSONObject(resBody);

                            if (resJson.has("prediction")) {
                                double prediction = resJson.getDouble("prediction");
                                String formattedPrediction = String.format("%.2f", prediction);

                                runOnUiThread(() -> {

                                    mainIntent.putExtra("soil_ph", soilPhInput.getText().toString());
                                    mainIntent.putExtra("soil_quality", soilQualityInput.getText().toString());
                                    mainIntent.putExtra("soil_temp", soilTempInput.getText().toString());
                                    mainIntent.putExtra("soil_humidity", soilHumidityInput.getText().toString());
                                    mainIntent.putExtra("wind_speed", windSpeedInput.getText().toString());
                                    mainIntent.putExtra("nitrogen", nitrogenInput.getText().toString());
                                    mainIntent.putExtra("phosphorus", phosphorusInput.getText().toString());
                                    mainIntent.putExtra("potassium", potassiumInput.getText().toString());
                                    mainIntent.putExtra("rainfall", rainfallInput.getText().toString());
                                    mainIntent.putExtra("crop_type", cropType);
                                    mainIntent.putExtra("soil_type", soilType);
                                    mainIntent.putExtra("prediction", formattedPrediction);

                                    predictionReady = true;
                                    checkAndLaunch();
                                });
                            } else {
                                runOnUiThread(() -> translationHelper.showTranslatedToast("Error: " + resJson.optString("error", "Unknown error")));
                            }
                        } catch (Exception e) {
                            runOnUiThread(() -> translationHelper.showTranslatedToast("Parsing Error"));
                        }
                    }
                }
            });

        } catch (Exception e) {
            translationHelper.showTranslatedToast("Input Error: " + e.getMessage());
        }
    }

    private void sendAgentRequest() {
        try {
            JSONObject json = new JSONObject();
            json.put("crop", spinnerCropType.getSelectedItem().toString().toLowerCase());
            json.put("soil_ph", Double.parseDouble(soilPhInput.getText().toString()));
            json.put("soil_moisture", Double.parseDouble(soilHumidityInput.getText().toString()));
            json.put("n", Double.parseDouble(nitrogenInput.getText().toString()));
            json.put("p", Double.parseDouble(phosphorusInput.getText().toString()));
            json.put("k", Double.parseDouble(potassiumInput.getText().toString()));

            RequestBody body = RequestBody.create(json.toString(), JSON);
            Request request = new Request.Builder()
                    .url("https://agriagent-final.onrender.com/analyze")
                    .post(body)
                    .build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    runOnUiThread(() -> translationHelper.showTranslatedToast("Agent API Error: " + e.getMessage()));
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (!response.isSuccessful()) {
                        runOnUiThread(() -> translationHelper.showTranslatedToast("Agent Server Error: " + response.code()));
                    } else {
                        try {
                            String resBody = response.body().string();
                            JSONObject resJson = new JSONObject(resBody);

                            JSONObject optimized = resJson.optJSONObject("optimized_values");
                            String recommendations = resJson.optString("recommendations", "No recommendations available");

                            runOnUiThread(() -> {
                                if (optimized != null) {
                                    double optPh = optimized.optDouble("soil_ph", Double.NaN);
                                    double optMoisture = optimized.optDouble("soil_moisture", Double.NaN);
                                    double optN = optimized.optDouble("n", Double.NaN);
                                    double optP = optimized.optDouble("p", Double.NaN);
                                    double optK = optimized.optDouble("k", Double.NaN);

                                    mainIntent.putExtra("opt_soil_ph", optPh);
                                    mainIntent.putExtra("opt_soil_moisture", optMoisture);
                                    mainIntent.putExtra("opt_n", optN);
                                    mainIntent.putExtra("opt_p", optP);
                                    mainIntent.putExtra("opt_k", optK);
                                    mainIntent.putExtra("recommendations", recommendations);

                                    sendOptimizedPredictionRequest(optN, optP, optK, optPh, optMoisture);

                                    agentReady = true;
                                } else {
                                    translationHelper.showTranslatedToast("No optimized values found");
                                    agentReady = true;
                                    checkAndLaunch();
                                }
                            });
                        } catch (Exception e) {
                            runOnUiThread(() -> translationHelper.showTranslatedToast("Agent Parsing Error: " + e.getMessage()));
                        }
                    }
                }
            });

        } catch (Exception e) {
            translationHelper.showTranslatedToast("Agent Input Error: " + e.getMessage());
        }
    }

    private void sendOptimizedPredictionRequest(double optN, double optP, double optK, double optPh, double optMoisture) {
        try {
            String cropType = spinnerCropType.getSelectedItem().toString();
            String soilType = spinnerSoilType.getSelectedItem().toString();

            JSONObject json = new JSONObject();
            json.put("crop_type", cropType);
            json.put("soil_type", soilType);
            json.put("soil_ph", optPh);
            json.put("temperature", Double.parseDouble(soilTempInput.getText().toString()));
            json.put("humidity", optMoisture);
            json.put("wind_speed", Double.parseDouble(windSpeedInput.getText().toString()));
            json.put("n", optN);
            json.put("p", optP);
            json.put("k", optK);
            json.put("soil_quality", Double.parseDouble(soilQualityInput.getText().toString()));
            json.put("avg_rainfall", Double.parseDouble(rainfallInput.getText().toString()));

            RequestBody body = RequestBody.create(json.toString(), JSON);
            Request request = new Request.Builder().url(BASE_URL).post(body).build();

            client.newCall(request).enqueue(new Callback() {
                @Override
                public void onFailure(Call call, IOException e) {
                    runOnUiThread(() -> translationHelper.showTranslatedToast("Optimized API Error: " + e.getMessage()));
                }

                @Override
                public void onResponse(Call call, Response response) throws IOException {
                    if (!response.isSuccessful()) {
                        runOnUiThread(() -> translationHelper.showTranslatedToast("Optimized Server Error: " + response.code()));
                    } else {
                        try {
                            String resBody = response.body().string();
                            JSONObject resJson = new JSONObject(resBody);

                            if (resJson.has("prediction")) {
                                double optimizedYield = resJson.getDouble("prediction");
                                String formattedOptimizedYield = String.format("%.2f", optimizedYield);

                                runOnUiThread(() -> {
                                    mainIntent.putExtra("optimized_yield", formattedOptimizedYield);

                                    // Only start result once
                                    checkAndLaunch();
                                });
                            }
                        } catch (Exception e) {
                            runOnUiThread(() -> translationHelper.showTranslatedToast("Optimized Parsing Error"));
                        }
                    }
                }
            });

        } catch (Exception e) {
            translationHelper.showTranslatedToast("Optimized Input Error: " + e.getMessage());
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
