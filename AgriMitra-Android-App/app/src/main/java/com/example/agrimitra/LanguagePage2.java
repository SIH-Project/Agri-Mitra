package com.example.agrimitra;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.View;
import android.widget.RadioButton;
import android.widget.RadioGroup;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class LanguagePage2 extends AppCompatActivity {

    private CardView getStarted;
    private RadioGroup radioGroupLanguages;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_language_page2);

        getStarted = findViewById(R.id.getStarted);
        radioGroupLanguages = findViewById(R.id.radioGroupLanguages);




        // Load saved language preference if exists
        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
        String savedLang = prefs.getString("SelectedLanguage", "en"); // default English
        selectSavedLanguage(savedLang);

        // Continue button click
        getStarted.setOnClickListener(v -> {
            String selectedLangCode = getSelectedLanguageCode();
            // Save language preference
            prefs.edit().putString("SelectedLanguage", selectedLangCode).apply();

            // Go to MainPage
            startActivity(new Intent(LanguagePage2.this, MainPage.class));
            finish();
        });

        // Handle window insets (Edge to Edge)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    // Get ML Kit / App language code based on selected radio button
    // Get ML Kit / App language code based on selected radio button
    private String getSelectedLanguageCode() {
        int selectedId = radioGroupLanguages.getCheckedRadioButtonId();
        RadioButton selectedButton = findViewById(selectedId);

        if (selectedButton == null) return "en";

        if (selectedButton.getId() == R.id.langEnglish) {
            return "en";
        } else if (selectedButton.getId() == R.id.langHindi) {
            return "hi";
        } else if (selectedButton.getId() == R.id.langTamil) {
            return "ta";
        } else if (selectedButton.getId() == R.id.langMalayalam) {
            return "ml";
        } else if (selectedButton.getId() == R.id.langKannada) {
            return "kn";
        } else if (selectedButton.getId() == R.id.langGujarati) {
            return "gu";
        } else if (selectedButton.getId() == R.id.langBengali) {
            return "bn";
        } else if (selectedButton.getId() == R.id.langMarathi) {
            return "mr";
        } else if (selectedButton.getId() == R.id.langTelugu) {
            return "te";
        } else {
            return "en";
        }
    }

    // Select the radio button based on saved language
    private void selectSavedLanguage(String langCode) {
        if (langCode.equals("en")) {
            radioGroupLanguages.check(R.id.langEnglish);
        } else if (langCode.equals("hi")) {
            radioGroupLanguages.check(R.id.langHindi);
        } else if (langCode.equals("ta")) {
            radioGroupLanguages.check(R.id.langTamil);
        } else if (langCode.equals("ml")) {
            radioGroupLanguages.check(R.id.langMalayalam);
        } else if (langCode.equals("kn")) {
            radioGroupLanguages.check(R.id.langKannada);
        } else if (langCode.equals("gu")) {
            radioGroupLanguages.check(R.id.langGujarati);
        } else if (langCode.equals("bn")) {
            radioGroupLanguages.check(R.id.langBengali);
        } else if (langCode.equals("mr")) {
            radioGroupLanguages.check(R.id.langMarathi);
        } else if (langCode.equals("te")) {
            radioGroupLanguages.check(R.id.langTelugu);
        } else {
            radioGroupLanguages.check(R.id.langEnglish);
        }
    }

}
