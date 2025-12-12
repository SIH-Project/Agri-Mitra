package com.example.agrimitra;

import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.speech.tts.TextToSpeech;
import android.widget.Toast;

import java.util.Locale;


public class CropRecommendationResultPage extends AppCompatActivity {

    private TranslationHelper translationHelper;

    TextView t2,t4,t6,t8,t10,t12,t14,t16,t18;

    private TextToSpeech textToSpeech;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_crop_recommendation_result_page);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        ImageView imageView=findViewById(R.id.img1);

        TextView title=findViewById(R.id.t1);
        TextView description=findViewById(R.id.t3);
        TextView pricePer=findViewById(R.id.t5);
        TextView avgProducing=findViewById(R.id.t7);
        TextView time=findViewById(R.id.t9);
        TextView profit=findViewById(R.id.t11);
        TextView msp=findViewById(R.id.t13);
        TextView season=findViewById(R.id.t15);
        TextView tools=findViewById(R.id.t17);
        TextView state=findViewById(R.id.t19);

        t2 = findViewById(R.id.t2);
        t4 = findViewById(R.id.t4);
        t6 = findViewById(R.id.t6);
        t8 = findViewById(R.id.t8);
        t10 = findViewById(R.id.t10);
        t12 = findViewById(R.id.t12);
        t14 = findViewById(R.id.t14);
        t16 = findViewById(R.id.t16);
        t18 = findViewById(R.id.t18);





        // Intent
        String crop=getIntent().getStringExtra("crop").toString().toLowerCase();

        // Initialize translation and set crop data
        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(t2, "Description");
            translationHelper.translateTextView(t4, "Market price per kg");
            translationHelper.translateTextView(t6, "Average producing cost/ha");
            translationHelper.translateTextView(t8, "Time to produce(crop cycle)");
            translationHelper.translateTextView(t10, "Estimated profit/ha");
            translationHelper.translateTextView(t12, "Minimum Support Price(MSPs)");
            translationHelper.translateTextView(t14, "Favourable season");
            translationHelper.translateTextView(t16, "Tools used");
            translationHelper.translateTextView(t18, "Most producing state");


            setCropData(crop, title, description, pricePer, avgProducing, time, profit, msp, season, tools, state);
        });


        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                // Default is English
                Locale ttsLocale = Locale.ENGLISH;

                switch (translationHelper.getSelectedLanguage()) {
                    case "ta": // Tamil
                        ttsLocale = new Locale("ta", "IN");
                        break;
                    case "en": // English
                    default:
                        ttsLocale = Locale.ENGLISH;
                        break;
                }

                int result = textToSpeech.setLanguage(ttsLocale);
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Toast.makeText(this, "TTS language not supported on this device", Toast.LENGTH_LONG).show();
                }

                textToSpeech.setPitch(1.0f);
                textToSpeech.setSpeechRate(1.0f);
            }
        });


        ImageButton explainButton = findViewById(R.id.explain);
        explainButton.setOnClickListener(v -> {
            if (textToSpeech != null) {

                String message = "";  // English base message

                switch (crop) {
                    case "rice":
                        message = "Your recommended crop is Rice. Description: Staple cereal for most of India; grown in flooded or puddled fields or upland in some areas. High calorie staple and market crop.";
                        break;
                    case "maize":
                        message = "Your recommended crop is Maize. Description: Versatile cereal used as food, feed and industrial raw material. High returns with hybrid varieties and irrigation.";
                        break;
                    case "chickpea":
                        message = "Your recommended crop is Chickpea. Description: Important pulse used as protein source; drought tolerant legumes fix some nitrogen.";
                        break;
                    case "kidneybeans":
                        message = "Your recommended crop is Kidney beans, also called Rajma. Description: Crop grown in some hill and plain regions; good protein crop.";
                        break;
                    case "pigeonpeas":
                        message = "Your recommended crop is Pigeon peas, also called Tur or Red gram. Description: Key pulse crop in semi-arid regions; important protein source and intercrop option.";
                        break;
                    case "mothbeans":
                        message = "Your recommended crop is Moth Beans. Description: Drought-resistant legume grown mainly in arid regions. Provides protein and is used for dal and fodder.";
                        break;
                    case "mungbean":
                        message = "Your recommended crop is Mung Bean, also called Green Gram. Description: Short-duration pulse crop, rich in protein and widely used as dal and sprouts. Improves soil fertility.";
                        break;
                    case "blackgram":
                        message = "Your recommended crop is Black Gram, also called Urad. Description: Popular pulse crop used in dals, idli/dosa batter, and papads. Improves soil nitrogen.";
                        break;
                    case "lentil":
                        message = "Your recommended crop is Lentil. Description: Winter pulse crop rich in protein. Consumed as dal and plays a role in crop rotation systems.";
                        break;
                    case "pomegranate":
                        message = "Your recommended crop is Pomegranate. Description: High-value fruit crop known for its medicinal and nutritional benefits. Export-oriented crop.";
                        break;
                    case "banana":
                        message = "Your recommended crop is Banana. Description: Tropical fruit crop and one of the largest produced fruits in India. Provides quick returns.";
                        break;
                    case "mango":
                        message = "Your recommended crop is Mango. Description: National fruit of India, highly profitable with export demand. Popular varieties include Alphonso and Dasheri.";
                        break;
                    case "grapes":
                        message = "Your recommended crop is Grapes. Description: Fruit crop with high market and export value. Requires good vineyard management and irrigation.";
                        break;
                    case "watermelon":
                        message = "Your recommended crop is Watermelon. Description: Summer fruit crop with high water content. Quick growing and widely consumed in hot climates.";
                        break;
                    case "muskmelon":
                        message = "Your recommended crop is Muskmelon. Description: Sweet melon fruit crop grown in warm climates. Popular in summer with high consumer demand.";
                        break;
                    case "apple":
                        message = "Your recommended crop is Apple. Description: Popular temperate fruit crop, majorly grown in hilly regions. Requires cold climate.";
                        break;
                    case "orange":
                        message = "Your recommended crop is Orange. Description: Citrus fruit rich in Vitamin C. India is among the largest producers, with Nagpur famous for oranges.";
                        break;
                    case "papaya":
                        message = "Your recommended crop is Papaya. Description: Fast-growing fruit crop with high demand. Rich in vitamins and used for fresh eating and processing.";
                        break;
                    case "coconut":
                        message = "Your recommended crop is Coconut. Description: Major plantation crop of coastal India, providing copra, oil, and other products. Lifeline of coastal farmers.";
                        break;
                    case "cotton":
                        message = "Your recommended crop is Cotton. Description: Cash crop and primary source of natural fiber. Used in textile industry worldwide.";
                        break;
                    case "jute":
                        message = "Your recommended crop is Jute. Description: Important fiber crop known as the 'golden fiber'. Used for making gunny bags, ropes, and mats.";
                        break;
                    case "coffee":
                        message = "Your recommended crop is Coffee. Description: Plantation crop grown in hilly areas. India mainly produces Arabica and Robusta varieties.";
                        break;
                    default:
                        message = "Your recommended crop is Rice. Description: Staple cereal for most of India; grown in flooded or puddled fields or upland in some areas. High calorie staple and market crop.";
                        break;
                }

                // ✅ Now handle multilingual TTS
                String selectedLang = translationHelper.getSelectedLanguage();

                if ("ta".equals(selectedLang)) {
                    // Translate English message to Tamil first
                    translationHelper.translateText(message, translatedText -> {
                        textToSpeech.setLanguage(new Locale("ta", "IN"));
                        textToSpeech.speak(translatedText, TextToSpeech.QUEUE_FLUSH, null, "tts_ta");
                    });
                } else {
                    // Default: English
                    textToSpeech.setLanguage(Locale.ENGLISH);
                    textToSpeech.speak(message, TextToSpeech.QUEUE_FLUSH, null, "tts_en");
                }
            }
        });


























        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void setCropData(String crop, TextView title, TextView description, TextView pricePer,
                           TextView avgProducing, TextView time, TextView profit, TextView msp,
                           TextView season, TextView tools, TextView state) {

        if (crop.equals("rice")) {
            translationHelper.translateTextView(title, "Rice");
            translationHelper.translateTextView(description, "Staple cereal for most of India; grown in flooded/puddled fields or upland in some areas. High calorie staple and staple-food market crop.");
            pricePer.setText("₹40–₹60");
            avgProducing.setText("₹50,000 – ₹90,000");
            time.setText("120–150 days");
            profit.setText("₹10,000–₹40,000");
            msp.setText("₹23.69/kg");
            translationHelper.translateTextView(season, "Kharif (June–Nov)");
            translationHelper.translateTextView(tools, "Tractor, rotavator, puddler/transplanter, threshers, winnowers.");
            translationHelper.translateTextView(state, "West Bengal");
        }
        else if (crop.equals("maize")) {
            translationHelper.translateTextView(title, "Maize");
            translationHelper.translateTextView(description, "Versatile cereal used as food, feed and industrial raw material. High returns with hybrid varieties and irrigation.");
            pricePer.setText("₹18–₹30");
            avgProducing.setText("₹30,000 – ₹60,000");
            time.setText("90–120 days");
            profit.setText("₹10,000–₹40,000");
            msp.setText("₹23.69/kg");
            translationHelper.translateTextView(season, "Kharif (monsoon)");
            translationHelper.translateTextView(tools, "Tractor, seed drill/planter, cultivator, combine (for large farms), sheller.");
            translationHelper.translateTextView(state, "Karnataka");
        }
        else if (crop.equals("chickpea")) {
            translationHelper.translateTextView(title, "Chickpea");
            translationHelper.translateTextView(description, "Important pulse (gram) used as protein source; drought tolerant legumes fix some nitrogen.");
            pricePer.setText("₹60–₹110");
            avgProducing.setText("₹25,000 – ₹45,000");
            time.setText("90–120 days");
            profit.setText("₹8,000–₹30,000");
            msp.setText("₹23.69/kg");
            translationHelper.translateTextView(season, "Rabi (Oct–Mar)");
            translationHelper.translateTextView(tools, "Plough, seed drill, combine harvester for larger farms.");
            translationHelper.translateTextView(state, "Madhya Pradesh");
        }
        else if (crop.equals("kidneybeans")) {
            translationHelper.translateTextView(title, "Kidney beans (Rajma)");
            translationHelper.translateTextView(description, "Crop grown in some hill and plain regions; good protein crop.");
            translationHelper.translateTextView(state, "Jammu & Kashmir");
            time.setText("90–150 days");
            avgProducing.setText("₹30,000 – ₹60,000");
            pricePer.setText("₹80–₹140");
            translationHelper.translateTextView(tools, "Plough, seed drill, manual harvest or small combine.");
            translationHelper.translateTextView(season, "Kharif/Rabi");
            profit.setText("₹5,000–₹30,000");
            msp.setText("₹23.69/kg");
        }
        else if (crop.equals("pigeonpeas")) {
            translationHelper.translateTextView(title, "Pigeon peas (Tur / Red gram)");
            translationHelper.translateTextView(description, "Key pulse crop in semi-arid regions; important protein source and intercrop option.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("120–160 days");
            avgProducing.setText("₹20,000 – ₹45,000");
            pricePer.setText("₹70–₹140");
            translationHelper.translateTextView(tools, "Plough, seed drill, combine (where mechanized)");
            translationHelper.translateTextView(season, "Kharif");
            profit.setText("₹5,000–₹25,000");
            msp.setText("₹23.69/kg");
        }
        else if (crop.equals("mothbeans")) {
            translationHelper.translateTextView(title, "Moth Beans");
            translationHelper.translateTextView(description, "A drought-resistant legume grown mainly in arid regions of India. Provides protein and is used for dal and fodder.");
            translationHelper.translateTextView(state, "Rajasthan");
            time.setText("70–90 days");
            avgProducing.setText("₹15,000 – ₹35,000 per hectare");
            pricePer.setText("₹60–₹90 per kg");
            translationHelper.translateTextView(tools, "Plough, seed drill, hand weeding tools");
            translationHelper.translateTextView(season, "Kharif (July–October)");
            profit.setText("₹8,000 – ₹20,000 per hectare");
            translationHelper.translateTextView(msp, "No fixed MSP (market price based)");
        }
        else if (crop.equals("mungbean")) {
            translationHelper.translateTextView(title, "Mung Bean (Green Gram)");
            translationHelper.translateTextView(description, "Short-duration pulse crop, rich in protein and widely used as dal and sprouts. Improves soil fertility.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("60–75 days");
            avgProducing.setText("₹20,000 – ₹40,000 per hectare");
            pricePer.setText("₹80–₹120 per kg");
            translationHelper.translateTextView(tools, "Seed drill, plough, threshing tools");
            translationHelper.translateTextView(season, "Kharif & Summer");
            profit.setText("₹10,000 – ₹25,000 per hectare");
            msp.setText("₹86.82/kg (MSP 2024–25)");
        }
        else if (crop.equals("blackgram")) {
            translationHelper.translateTextView(title, "Black Gram (Urad)");
            translationHelper.translateTextView(description, "Popular pulse crop used in dals, idli/dosa batter, and papads. Improves soil nitrogen.");
            translationHelper.translateTextView(state, "Madhya Pradesh");
            time.setText("70–90 days");
            avgProducing.setText("₹18,000 – ₹40,000 per hectare");
            pricePer.setText("₹70–₹100 per kg");
            translationHelper.translateTextView(tools, "Plough, seed drill, harvesting sickles");
            translationHelper.translateTextView(season, "Kharif & Summer");
            profit.setText("₹8,000 – ₹20,000 per hectare");
            msp.setText("₹74.00/kg (MSP 2025–26)");
        }
        else if (crop.equals("lentil")) {
            translationHelper.translateTextView(title, "Lentil");
            translationHelper.translateTextView(description, "Winter pulse crop rich in protein. Consumed as dal and plays a role in crop rotation systems.");
            translationHelper.translateTextView(state, "Uttar Pradesh");
            time.setText("100–120 days");
            avgProducing.setText("₹25,000 – ₹50,000 per hectare");
            pricePer.setText("₹60–₹90 per kg");
            translationHelper.translateTextView(tools, "Plough, seed drill, harrow, sickles");
            translationHelper.translateTextView(season, "Rabi");
            profit.setText("₹12,000 – ₹25,000 per hectare");
            msp.setText("₹65.00/kg (approx MSP)");
        }
        else if (crop.equals("pomegranate")) {
            translationHelper.translateTextView(title, "Pomegranate");
            translationHelper.translateTextView(description, "A high-value fruit crop known for its medicinal and nutritional benefits. Export-oriented crop.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("150–180 days (flower to harvest)");
            avgProducing.setText("₹80,000 – ₹1,50,000 per hectare");
            pricePer.setText("₹60–₹120 per kg");
            translationHelper.translateTextView(tools, "Pruning tools, drip irrigation, sprayers");
            translationHelper.translateTextView(season, "All year (best in winter)");
            profit.setText("₹40,000 – ₹90,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("banana")) {
            translationHelper.translateTextView(title, "Banana");
            translationHelper.translateTextView(description, "Tropical fruit crop and one of the largest produced fruits in India. Provides quick returns.");
            translationHelper.translateTextView(state, "Tamil Nadu");
            time.setText("9–12 months");
            avgProducing.setText("₹1,00,000 – ₹2,00,000 per hectare");
            pricePer.setText("₹10–₹25 per kg");
            translationHelper.translateTextView(tools, "Irrigation systems, cutting knives, sprayers");
            translationHelper.translateTextView(season, "Year-round");
            profit.setText("₹50,000 – ₹1,00,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("mango")) {
            translationHelper.translateTextView(title, "Mango");
            translationHelper.translateTextView(description, "National fruit of India, highly profitable with export demand. Popular varieties include Alphonso and Dasheri.");
            translationHelper.translateTextView(state, "Uttar Pradesh");
            time.setText("3–5 years (tree maturity)");
            avgProducing.setText("₹80,000 – ₹2,00,000 per hectare");
            pricePer.setText("₹30–₹80 per kg");
            translationHelper.translateTextView(tools, "Pruning tools, ladders, harvesting nets");
            translationHelper.translateTextView(season, "Summer (April–July)");
            profit.setText("₹50,000 – ₹1,50,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("grapes")) {
            translationHelper.translateTextView(title, "Grapes");
            translationHelper.translateTextView(description, "Fruit crop with high market and export value. Requires good vineyard management and irrigation.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("4–5 months (fruiting)");
            avgProducing.setText("₹1,50,000 – ₹3,00,000 per hectare");
            pricePer.setText("₹50–₹120 per kg");
            translationHelper.translateTextView(tools, "Pruning shears, trellis system, sprayers");
            translationHelper.translateTextView(season, "January–March (main harvest)");
            profit.setText("₹70,000 – ₹2,00,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("watermelon")) {
            translationHelper.translateTextView(title, "Watermelon");
            translationHelper.translateTextView(description, "Summer fruit crop with high water content. Quick growing and widely consumed in hot climates.");
            translationHelper.translateTextView(state, "Karnataka");
            time.setText("70–90 days");
            avgProducing.setText("₹30,000 – ₹60,000 per hectare");
            pricePer.setText("₹10–₹25 per kg");
            translationHelper.translateTextView(tools, "Plough, drip irrigation, knives for harvest");
            translationHelper.translateTextView(season, "Summer");
            profit.setText("₹15,000 – ₹35,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("muskmelon")) {
            translationHelper.translateTextView(title, "Muskmelon");
            translationHelper.translateTextView(description, "Sweet melon fruit crop grown in warm climates. Popular in summer with high consumer demand.");
            translationHelper.translateTextView(state, "Uttar Pradesh");
            time.setText("70–80 days");
            avgProducing.setText("₹25,000 – ₹55,000 per hectare");
            pricePer.setText("₹20–₹50 per kg");
            translationHelper.translateTextView(tools, "Plough, drip irrigation, hand tools");
            translationHelper.translateTextView(season, "Summer");
            profit.setText("₹10,000 – ₹30,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("apple")) {
            translationHelper.translateTextView(title, "Apple");
            translationHelper.translateTextView(description, "Popular temperate fruit crop, majorly grown in hilly regions. Requires cold climate.");
            translationHelper.translateTextView(state, "Himachal Pradesh");
            time.setText("7–8 months (flower to harvest)");
            avgProducing.setText("₹1,50,000 – ₹3,50,000 per hectare");
            pricePer.setText("₹60–₹150 per kg");
            translationHelper.translateTextView(tools, "Pruning tools, ladders, harvesting bags");
            translationHelper.translateTextView(season, "Autumn (September–October)");
            profit.setText("₹80,000 – ₹2,00,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("orange")) {
            translationHelper.translateTextView(title, "Orange");
            translationHelper.translateTextView(description, "Citrus fruit rich in Vitamin C. India is among the largest producers, with Nagpur famous for oranges.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("7–8 months");
            avgProducing.setText("₹80,000 – ₹1,80,000 per hectare");
            pricePer.setText("₹30–₹80 per kg");
            translationHelper.translateTextView(tools, "Pruning shears, sprayers, harvesting bags");
            translationHelper.translateTextView(season, "Winter (Dec–Feb)");
            profit.setText("₹40,000 – ₹1,00,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("papaya")) {
            translationHelper.translateTextView(title, "Papaya");
            translationHelper.translateTextView(description, "Fast-growing fruit crop with high demand. Rich in vitamins and used for fresh eating and processing.");
            translationHelper.translateTextView(state, "Andhra Pradesh");
            time.setText("8–10 months");
            avgProducing.setText("₹70,000 – ₹1,50,000 per hectare");
            pricePer.setText("₹15–₹40 per kg");
            translationHelper.translateTextView(tools, "Drip irrigation, cutting tools, sprayers");
            translationHelper.translateTextView(season, "Year-round");
            profit.setText("₹30,000 – ₹80,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("coconut")) {
            translationHelper.translateTextView(title, "Coconut");
            translationHelper.translateTextView(description, "Major plantation crop of coastal India, providing copra, oil, and other products. Lifeline of coastal farmers.");
            translationHelper.translateTextView(state, "Kerala");
            time.setText("12 months (continuous harvesting)");
            avgProducing.setText("₹1,00,000 – ₹2,50,000 per hectare");
            pricePer.setText("₹30–₹60 per nut equivalent");
            translationHelper.translateTextView(tools, "Climbing tools, knives, sprayers");
            translationHelper.translateTextView(season, "Year-round");
            profit.setText("₹60,000 – ₹1,50,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else if (crop.equals("cotton")) {
            translationHelper.translateTextView(title, "Cotton");
            translationHelper.translateTextView(description, "Cash crop and primary source of natural fiber. Used in textile industry worldwide.");
            translationHelper.translateTextView(state, "Maharashtra");
            time.setText("150–180 days");
            avgProducing.setText("₹40,000 – ₹90,000 per hectare");
            pricePer.setText("₹60–₹80 per kg (lint)");
            translationHelper.translateTextView(tools, "Seed drill, plough, cotton picker");
            translationHelper.translateTextView(season, "Kharif");
            profit.setText("₹15,000 – ₹40,000 per hectare");
            msp.setText("₹62–₹70/kg lint equivalent (MSP varies)");
        }
        else if (crop.equals("jute")) {
            translationHelper.translateTextView(title, "Jute");
            translationHelper.translateTextView(description, "Important fiber crop known as the 'golden fiber'. Used for making gunny bags, ropes, and mats.");
            translationHelper.translateTextView(state, "West Bengal");
            time.setText("120–150 days");
            avgProducing.setText("₹35,000 – ₹70,000 per hectare");
            pricePer.setText("₹40–₹60 per kg (fiber)");
            translationHelper.translateTextView(tools, "Plough, sickle, retting tanks");
            translationHelper.translateTextView(season, "Kharif (March–July sowing)");
            profit.setText("₹15,000 – ₹30,000 per hectare");
            msp.setText("₹47/kg fiber (approx MSP)");
        }
        else if (crop.equals("coffee")) {
            translationHelper.translateTextView(title, "Coffee");
            translationHelper.translateTextView(description, "Plantation crop grown in hilly areas. India mainly produces Arabica and Robusta varieties.");
            translationHelper.translateTextView(state, "Karnataka");
            time.setText("3–4 years (to start production)");
            avgProducing.setText("₹1,50,000 – ₹3,00,000 per hectare");
            pricePer.setText("₹120–₹200 per kg");
            translationHelper.translateTextView(tools, "Pruning tools, pulping machines, dryers");
            translationHelper.translateTextView(season, "Harvest: Nov–Feb");
            profit.setText("₹70,000 – ₹1,50,000 per hectare");
            translationHelper.translateTextView(msp, "No MSP (market driven)");
        }
        else {
            // Default to Rice
            translationHelper.translateTextView(title, "Rice");
            translationHelper.translateTextView(description, "Staple cereal for most of India; grown in flooded/puddled fields or upland in some areas. High calorie staple and staple-food market crop.");
            pricePer.setText("₹40–₹60");
            avgProducing.setText("₹50,000 – ₹90,000");
            time.setText("120–150 days");
            profit.setText("₹10,000–₹40,000");
            msp.setText("₹23.69/kg");
            translationHelper.translateTextView(season, "Kharif (June–Nov)");
            translationHelper.translateTextView(tools, "Tractor, rotavator, puddler/transplanter, threshers, winnowers.");
            translationHelper.translateTextView(state, "West Bengal");
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }

        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (translationHelper != null) {
            translationHelper.close();
        }


    }
}