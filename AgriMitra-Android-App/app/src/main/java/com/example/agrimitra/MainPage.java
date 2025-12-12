package com.example.agrimitra;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.fragment.app.Fragment;

import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainPage extends AppCompatActivity {

    private BottomNavigationView bottomNav;
    private String selectedLang;
    private MLKitTranslator translator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main_page);

        bottomNav = findViewById(R.id.bottom_navigation);

        // Load saved language
        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
        selectedLang = prefs.getString("SelectedLanguage", "en");

        // Initialize translator
        translator = new MLKitTranslator();
        translator.initTranslator("en", selectedLang);

        // Download model automatically and then translate menu
        translator.downloadModelIfNeeded(
                this::translateBottomNavMenu,
                () -> Log.e("MLKit", "Failed to download translation model")
        );

        // Load default fragment
        if (savedInstanceState == null) {
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.fragment_container, new CropPage())
                    .commit();
        }

        // BottomNavigationView item click handling (if-else)
        bottomNav.setOnItemSelectedListener(item -> {
            Fragment selected = null;
            if (item.getItemId() == R.id.crop_page) {
                selected = new CropPage();
            } else if (item.getItemId() == R.id.crop_page) {
                selected = new CommunityPage();
            } else if (item.getItemId() == R.id.crop_page) {
                selected = new PestPage();
            } else if (item.getItemId() == R.id.crop_page) {
                selected = new MarketPage();
            }

            if (selected != null) {
                getSupportFragmentManager().beginTransaction()
                        .replace(R.id.fragment_container, selected)
                        .commit();
            }
            return true;
        });

        // Edge-to-edge padding
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    // Translate BottomNavigationView menu items
    private void translateBottomNavMenu() {
        translateMenuItem(R.id.crop_page, "Home");
        translateMenuItem(R.id.community_page, "Simulator");
        translateMenuItem(R.id.pest_page, "Pests");
        translateMenuItem(R.id.market_page, "Market");
    }

    // Translate a single menu item
    private void translateMenuItem(int itemId, String text) {
        translator.translateText(text, translatedText -> {
            bottomNav.getMenu().findItem(itemId).setTitle(translatedText);
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translator != null) translator.close();
    }
}
