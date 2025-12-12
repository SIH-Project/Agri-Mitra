package com.example.agrimitra;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class YieldPredictionResult extends AppCompatActivity {

    private TranslationHelper translationHelper;

    // Declare all TextViews
    TextView stype, ctype, pH, temp, humi, wind, rain, n, p, k, pred, ohum, opH, on, op, ok, opred;
    TextView income, income1, income2, income3;

    // Extra textviews
    TextView t1, t2, t4, t6, t8, t10, t12;
    TextView t17, t14, t15, ot_t17, ot_t1, ot_t2, ot_t4, ot_t14, ot_t15, rt_t15,
            rt1_t15, rt2_t15, rt3_t15, rt4_t15, income_1, income_text1, income_text3, income_text5;

    double deltan,deltap,deltak;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_yield_prediction_result);

        translationHelper = new TranslationHelper(this);

        // Link all textviews
        stype = findViewById(R.id.t3);
        t1 = findViewById(R.id.t1);
        t2 = findViewById(R.id.t2);
        t4 = findViewById(R.id.t4);
        t6 = findViewById(R.id.t6);
        t8 = findViewById(R.id.t8);
        t10 = findViewById(R.id.t10);
        t12 = findViewById(R.id.t12);

        t17 = findViewById(R.id.t17);
        t14 = findViewById(R.id.t14);
        t15 = findViewById(R.id.t15);
        ot_t17 = findViewById(R.id.ot_t17);
        ot_t1 = findViewById(R.id.ot_t1);
        ot_t2 = findViewById(R.id.ot_t2);
        ot_t4 = findViewById(R.id.ot_t4);
        ot_t14 = findViewById(R.id.ot_t14);
        ot_t15 = findViewById(R.id.ot_t15);
        rt_t15 = findViewById(R.id.rt_t15);
        rt1_t15 = findViewById(R.id.rt1_t15);
        rt2_t15 = findViewById(R.id.rt2_t15);
        rt3_t15 = findViewById(R.id.rt3_t15);
        rt4_t15 = findViewById(R.id.rt4_t15);

        income_1 = findViewById(R.id.income1);
        income_text1 = findViewById(R.id.income_text1);
        income_text3 = findViewById(R.id.income_text3);
        income_text5 = findViewById(R.id.income_text5);

        ctype = findViewById(R.id.t4);
        pH = findViewById(R.id.t9);
        temp = findViewById(R.id.t7);
        humi = findViewById(R.id.t11);
        wind = findViewById(R.id.t13);
        rain = findViewById(R.id.t18);
        n = findViewById(R.id.n_value);
        p = findViewById(R.id.p_value);
        k = findViewById(R.id.k_value);
        pred = findViewById(R.id.t16);

        ohum = findViewById(R.id.ot_t3);
        opH = findViewById(R.id.ot_t5);
        on = findViewById(R.id.ot_n_value);
        op = findViewById(R.id.ot_p_value);
        ok = findViewById(R.id.ot_k_value);
        opred = findViewById(R.id.ot_t16);



        income1 = findViewById(R.id.income_text2);
        income2 = findViewById(R.id.income_text4);
        income3 = findViewById(R.id.income_text6);

        // Fetch data from intent
        String soilPh = getIntent().getStringExtra("soil_ph");
        String soilQuality = getIntent().getStringExtra("soil_quality");
        String soilTemp = getIntent().getStringExtra("soil_temp");
        String soilHumidity = getIntent().getStringExtra("soil_humidity");
        String windSpeed = getIntent().getStringExtra("wind_speed");
        String nitrogen = getIntent().getStringExtra("nitrogen");
        String phosphorus = getIntent().getStringExtra("phosphorus");
        String potassium = getIntent().getStringExtra("potassium");
        String rainfall = getIntent().getStringExtra("rainfall");


        double phx=Double.parseDouble(soilPh);
        double nx=Double.parseDouble(nitrogen);
        double phosx=Double.parseDouble(phosphorus);
        double px=Double.parseDouble(potassium);


        String cropType = getIntent().getStringExtra("crop_type");
        String soilType = getIntent().getStringExtra("soil_type");
        String prediction = getIntent().getStringExtra("prediction");

        double optN;
        optN=nx+5.8;
        double optP;
        optP=phosx+2.7;
        double optK;
        optK=px+9.8;
        double optPh = getIntent().getDoubleExtra("opt_soil_ph", Double.NaN);

        deltan=optN-nx;
        deltap=optP-phosx;
        deltak=optK-px;



        String optYield = getIntent().getStringExtra("optimized_yield");
        double x=Double.parseDouble(prediction);
        double y=x+17.8;
        String z=String.valueOf(y);


        // Apply translations
        translationHelper.initializeTranslation(() -> {
            // Labels
            translationHelper.translateTextView(t1, "Current Soil Details");
            translationHelper.translateTextView(t2, "Soil Type");
            translationHelper.translateTextView(t4, "Crop Type");
            translationHelper.translateTextView(t6, "Temperature");
            translationHelper.translateTextView(t8, "pH");
            translationHelper.translateTextView(t10, "Moisture");
            translationHelper.translateTextView(t12, "Windspeed");
            translationHelper.translateTextView(ot_t17, "Optimized Yield");

            translationHelper.translateTextView(ot_t1, "Optimized Property Values");
            translationHelper.translateTextView(ot_t2, "Moisture");
            translationHelper.translateTextView(ot_t4, "pH");
            translationHelper.translateTextView(ot_t14, "NPK values(in ppm)");
            translationHelper.translateTextView(ot_t15, "Optimized Yield");










            translationHelper.translateTextView(t17, "Average Rainfall");
            translationHelper.translateTextView(t14, "NPK values(in ppm)");
            translationHelper.translateTextView(t15, "Predicted Yield");

            translationHelper.translateTextView(rt_t15, "Adjust Nitrogen value");
            translationHelper.translateTextView(rt1_t15, "Alter Phosphorus value");
            translationHelper.translateTextView(rt2_t15, "Modify Potassium value");
            translationHelper.translateTextView(rt3_t15, "Change Soil pH value");
            translationHelper.translateTextView(rt4_t15, "Improve Water Moisture");

            translationHelper.translateTextView(income_1, "Optimized Income");
            translationHelper.translateTextView(income_text1, "Total Revenue");
            translationHelper.translateTextView(income_text3, "Expenditure Details");
            translationHelper.translateTextView(income_text5, "Net Profit");

            // Dynamic values
            translationHelper.translateTextView(stype, soilType);
            translationHelper.translateTextView(ctype, cropType);
            translationHelper.translateTextView(pH, soilPh);
            translationHelper.translateTextView(temp, soilTemp);
            translationHelper.translateTextView(humi, soilHumidity);
            translationHelper.translateTextView(wind, windSpeed);
            translationHelper.translateTextView(rain, rainfall);
            translationHelper.translateTextView(n, nitrogen);
            translationHelper.translateTextView(p, phosphorus);
            translationHelper.translateTextView(k, potassium);
            translationHelper.translateTextView(pred, "1.96"+" tons/acre");

            // Other (dummy values for now)
            translationHelper.translateTextView(ohum, "82.3");
            translationHelper.translateTextView(opH, "7.09");
            translationHelper.translateTextView(on, "82.5");
            translationHelper.translateTextView(op, "67.8");
            translationHelper.translateTextView(ok, "59.7");

            opH.setText(optPh+"");
            on.setText(optN+"");
            op.setText(optP+"");
            ok.setText(optK+"");






            translationHelper.translateTextView(opred, "2.23"+" tons/acre");

            // Example income calculation
            double produce = 2230;      // kg/acre
            double pricePerKg = 42.3;   // Rs./kg
            double revenue = produce * pricePerKg;

            double landPrepCost = 8000;
            double seedCost = 7000;
            double fertilizerCost = 25000;
            double pesticideCost = 10000;
            double laborCost = 30000;
            double irrigationCost = 1200;
            double postHarvestCost = 1000;

            double totalExpenditure = landPrepCost + seedCost + fertilizerCost + pesticideCost + laborCost + irrigationCost + postHarvestCost;
            double netProfit = revenue - totalExpenditure;

            translationHelper.translateTextView(income1,
                    "Production: " + produce + " Kg/acre\n" +
                            "Market Price (" + cropType + "): Rs. " + pricePerKg + "/Kg\n" +
                            "Revenue: Rs. " + String.format("%.2f",revenue) + "\n\n"
            );

            translationHelper.translateTextView(income2,
                    "Land Preparation: Rs. " + landPrepCost + "\n" +
                            "Seeds: Rs. " + seedCost + "\n" +
                            "Fertilizers: Rs. " + fertilizerCost + "\n" +
                            "Pesticides: Rs. " + pesticideCost + "\n" +
                            "Labor: Rs. " + laborCost + "\n" +
                            "Irrigation: Rs. " + irrigationCost + "\n" +
                            "Post-Harvest: Rs. " + postHarvestCost + "\n" +
                            "Total Expenditure: Rs. " + totalExpenditure + "\n\n"
            );

            translationHelper.translateTextView(income3, "Profit: Rs. " + String.format("%.2f",netProfit));
        });

        // Card click listeners
        setupRecommendationCards();

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void setupRecommendationCards() {
        CardView c1 = findViewById(R.id.rt_cv2);
        CardView c2 = findViewById(R.id.rt1_cv2);
        CardView c3 = findViewById(R.id.rt2_cv2);
        CardView c4 = findViewById(R.id.rt3_cv2);
        CardView c5 = findViewById(R.id.rt4_cv2);

        c1.setOnClickListener(v -> openRecommendation("nitrogen"));
        c2.setOnClickListener(v -> openRecommendation("phosphorus"));
        c3.setOnClickListener(v -> openRecommendation("potassium"));
        c4.setOnClickListener(v -> openRecommendation("ph"));
        c5.setOnClickListener(v -> openRecommendation("moisture"));
    }

    private void openRecommendation(String topic) {
        Intent intent = new Intent(YieldPredictionResult.this, RecommendationPage.class);
        intent.putExtra("topic", topic);
        intent.putExtra("budget", "20000");
        double delta=1.0;
        if (topic.equalsIgnoreCase("nitrogen")) delta=deltan;
        else if (topic.equalsIgnoreCase("phosphorus")) delta=deltap;
        else delta=deltak;
        intent.putExtra("delta",delta);
        startActivity(intent);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
