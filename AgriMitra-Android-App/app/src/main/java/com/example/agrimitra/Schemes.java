package com.example.agrimitra;

import android.os.Bundle;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class Schemes extends AppCompatActivity {

    CardView btnApplyPmKisan, btnApplyPmKusum, btnApplyFpo, btnApplyAif,
            btnApplySoilHealth, btnApplyPkvy, btnApplyMiss, btnApplyDigitalAgri,
            btnApplyNbhm, btnApplyMisPss;

    TextView a1,b1,c1,d1,
            a2,b2,c2,d2,
            a3,b3,c3,d3,
            a4,b4,c4,d4,
            a5,b5,c5,d5,
            a6,b6,c6,d6,
            a7,b7,c7,d7,
            a8,b8,c8,d8,
            a9,b9,c9,d9,
            a10,b10,c10,d10;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_schemes);


        a1 = findViewById(R.id.a1);
        b1 = findViewById(R.id.b1);
        c1 = findViewById(R.id.c1);
        d1 = findViewById(R.id.d1);

        // ------------------- CARD 2 -------------------
        a2 = findViewById(R.id.a2);
        b2 = findViewById(R.id.b2);
        c2 = findViewById(R.id.c2);
        d2 = findViewById(R.id.d2);

        // ------------------- CARD 3 -------------------
        a3 = findViewById(R.id.a3);
        b3 = findViewById(R.id.b3);
        c3 = findViewById(R.id.c3);
        d3 = findViewById(R.id.d3);

        // ------------------- CARD 4 -------------------
        a4 = findViewById(R.id.a4);
        b4 = findViewById(R.id.b4);
        c4 = findViewById(R.id.c4);
        d4 = findViewById(R.id.d4);

        // ------------------- CARD 5 -------------------
        a5 = findViewById(R.id.a5);
        b5 = findViewById(R.id.b5);
        c5 = findViewById(R.id.c5);
        d5 = findViewById(R.id.d5);

        // ------------------- CARD 6 -------------------
        a6 = findViewById(R.id.a6);
        b6 = findViewById(R.id.b6);
        c6 = findViewById(R.id.c6);
        d6 = findViewById(R.id.d6);

        // ------------------- CARD 7 -------------------
        a7 = findViewById(R.id.a7);
        b7 = findViewById(R.id.b7);
        c7 = findViewById(R.id.c7);
        d7 = findViewById(R.id.d7);

        // ------------------- CARD 8 -------------------
        a8 = findViewById(R.id.a8);
        b8 = findViewById(R.id.b8);
        c8 = findViewById(R.id.c8);
        d8 = findViewById(R.id.d8);

        // ------------------- CARD 9 -------------------
        a9 = findViewById(R.id.a9);
        b9 = findViewById(R.id.b9);
        c9 = findViewById(R.id.c9);
        d9 = findViewById(R.id.d9);

        // ------------------- CARD 10 -------------------
        a10 = findViewById(R.id.a10);
        b10 = findViewById(R.id.b10);
        c10 = findViewById(R.id.c10);
        d10 = findViewById(R.id.d10);

        TranslationHelper translationHelper=new TranslationHelper(this);


        btnApplyPmKisan = findViewById(R.id.btn_apply_pm_kisan);
        btnApplyPmKusum = findViewById(R.id.btn_apply_pm_kusum);
        btnApplyFpo = findViewById(R.id.btn_apply_fpo);
        btnApplyAif = findViewById(R.id.btn_apply_aif);
        btnApplySoilHealth = findViewById(R.id.btn_apply_soil_health);
        btnApplyPkvy = findViewById(R.id.btn_apply_pkvy);
        btnApplyMiss = findViewById(R.id.btn_apply_miss);
        btnApplyDigitalAgri = findViewById(R.id.btn_apply_digital_agriculture);
        btnApplyNbhm = findViewById(R.id.btn_apply_nbhm);
        btnApplyMisPss = findViewById(R.id.btn_apply_mis_pss);

        // Attach click listeners
        btnApplyPmKisan.setOnClickListener(v -> openUrl("https://pmkisan.gov.in/"));
        btnApplyPmKusum.setOnClickListener(v -> openUrl("https://www.india.gov.in/spotlight/pm-kusum-pradhan-mantri-kisan-urja-suraksha-evam-utthaan-mahabhiyan-scheme"));
        btnApplyFpo.setOnClickListener(v -> openUrl("https://www.nabard.org/"));
        btnApplyAif.setOnClickListener(v -> openUrl("https://agriinfra.dac.gov.in/Home"));
        btnApplySoilHealth.setOnClickListener(v -> openUrl("https://soilhealth.dac.gov.in/home"));
        btnApplyPkvy.setOnClickListener(v -> openUrl("https://agri.odisha.gov.in/node/193837"));
        btnApplyMiss.setOnClickListener(v -> openUrl("https://vajiramandravi.com/current-affairs/modified-interest-subvention-scheme/"));
        btnApplyDigitalAgri.setOnClickListener(v -> openUrl("https://agriwelfare.gov.in/en/DigiAgriDiv"));
        btnApplyNbhm.setOnClickListener(v -> openUrl("https://nbb.gov.in/default.html"));
        btnApplyMisPss.setOnClickListener(v -> openUrl("https://www.nafed-india.com/"));


        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(a1, a1.getText().toString());
            translationHelper.translateTextView(b1, b1.getText().toString());
            translationHelper.translateTextView(c1, c1.getText().toString());
            translationHelper.translateTextView(d1, d1.getText().toString());

            translationHelper.translateTextView(a2, a2.getText().toString());
            translationHelper.translateTextView(b2, b2.getText().toString());
            translationHelper.translateTextView(c2, c2.getText().toString());
            translationHelper.translateTextView(d2, d2.getText().toString());

            translationHelper.translateTextView(a3, a3.getText().toString());
            translationHelper.translateTextView(b3, b3.getText().toString());
            translationHelper.translateTextView(c3, c3.getText().toString());
            translationHelper.translateTextView(d3, d3.getText().toString());

            translationHelper.translateTextView(a4, a4.getText().toString());
            translationHelper.translateTextView(b4, b4.getText().toString());
            translationHelper.translateTextView(c4, c4.getText().toString());
            translationHelper.translateTextView(d4, d4.getText().toString());

            translationHelper.translateTextView(a5, a5.getText().toString());
            translationHelper.translateTextView(b5, b5.getText().toString());
            translationHelper.translateTextView(c5, c5.getText().toString());
            translationHelper.translateTextView(d5, d5.getText().toString());

            translationHelper.translateTextView(a6, a6.getText().toString());
            translationHelper.translateTextView(b6, b6.getText().toString());
            translationHelper.translateTextView(c6, c6.getText().toString());
            translationHelper.translateTextView(d6, d6.getText().toString());

            translationHelper.translateTextView(a7, a7.getText().toString());
            translationHelper.translateTextView(b7, b7.getText().toString());
            translationHelper.translateTextView(c7, c7.getText().toString());
            translationHelper.translateTextView(d7, d7.getText().toString());

            translationHelper.translateTextView(a8, a8.getText().toString());
            translationHelper.translateTextView(b8, b8.getText().toString());
            translationHelper.translateTextView(c8, c8.getText().toString());
            translationHelper.translateTextView(d8, d8.getText().toString());

            translationHelper.translateTextView(a9, a9.getText().toString());
            translationHelper.translateTextView(b9, b9.getText().toString());
            translationHelper.translateTextView(c9, c9.getText().toString());
            translationHelper.translateTextView(d9, d9.getText().toString());

            translationHelper.translateTextView(a10, a10.getText().toString());
            translationHelper.translateTextView(b10, b10.getText().toString());
            translationHelper.translateTextView(c10, c10.getText().toString());
            translationHelper.translateTextView(d10, d10.getText().toString());



        });






        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }


    private void openUrl(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        startActivity(intent);
    }


}