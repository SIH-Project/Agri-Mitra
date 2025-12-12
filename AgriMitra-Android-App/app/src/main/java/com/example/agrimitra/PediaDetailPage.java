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
import java.util.Locale;



public class PediaDetailPage extends AppCompatActivity {

    private TranslationHelper translationHelper;

    TextView t2,t4,t6;
    private TextToSpeech textToSpeech;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_pedia_detail_page);

        translationHelper = new TranslationHelper(this);

        String topic = getIntent().getStringExtra("topic").toLowerCase();

        ImageView imageView = findViewById(R.id.img1);
        TextView title = findViewById(R.id.t1);
        TextView description = findViewById(R.id.t3);
        TextView keyIngredients = findViewById(R.id.t5);
        TextView source = findViewById(R.id.t7);

        t2 = findViewById(R.id.t2);
        t4 = findViewById(R.id.t4);
        t6 = findViewById(R.id.t6); // Make sure this exists in your XML


        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(t2, "Description");
            translationHelper.translateTextView(t4, "Key Ingredients");
            translationHelper.translateTextView(t6, "Source"); // Replace with actual text
        });

        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.ENGLISH);
                textToSpeech.setPitch(1.0f);
                textToSpeech.setSpeechRate(1.0f);
            }
        });


        ImageButton explainButton = findViewById(R.id.explain);
        explainButton.setOnClickListener(v -> {
            // Speak the description when clicked
            speakDescription(description.getText().toString());
        });















        translationHelper.initializeTranslation(() -> {
            switch (topic) {

                case "compost":
                    imageView.setImageResource(R.drawable.compost);
                    translationHelper.translateTextView(title, "Compost");
                    translationHelper.translateTextView(description, "Compost is an organic soil amendment made from decomposed plant and animal waste. It improves soil fertility, structure, and water retention, while providing essential nutrients for crops. Compost also enhances microbial activity, which helps plants absorb nutrients more efficiently. Using compost reduces the need for chemical fertilizers, making farming more sustainable. Regular application improves long-term soil health, increases yields, and prevents erosion.");
                    translationHelper.translateTextView(keyIngredients, "Decomposed crop residues, kitchen waste, animal dung, leaves.");
                    translationHelper.translateTextView(source, "Natural (biodegradable organic matter).");
                    break;

                case "animal manure":
                    imageView.setImageResource(R.drawable.manure);
                    translationHelper.translateTextView(title, "Animal Manure");
                    translationHelper.translateTextView(description, "Animal manure is a traditional organic fertilizer obtained from livestock waste. It enriches soil with essential nutrients like nitrogen, phosphorus, and potassium while improving soil texture. It boosts soil microbes and helps retain moisture. Different animals (cow, goat, poultry) provide manure with varying nutrient contents. When applied properly, it increases crop yields and reduces the need for chemical fertilizers. However, it should be well-composted to avoid harmful pathogens.");
                    translationHelper.translateTextView(keyIngredients, "Nitrogen, phosphorus, potassium, organic matter.");
                    translationHelper.translateTextView(source, "Natural (animal waste).");
                    break;

                case "green manure":
                    imageView.setImageResource(R.drawable.gm);
                    translationHelper.translateTextView(title, "Green Manure");
                    translationHelper.translateTextView(description, "Green manure involves growing specific crops (like legumes, clover, or sunhemp) and plowing them back into the soil before maturity. These crops add organic matter, improve soil structure, and enrich nitrogen levels through biological nitrogen fixation. Green manures prevent weed growth, reduce soil erosion, and enhance soil fertility naturally.");
                    translationHelper.translateTextView(keyIngredients, "Nitrogen-rich legume plants (sunhemp, clover, beans).");
                    translationHelper.translateTextView(source, "Natural (cultivated plants grown for soil enrichment).");
                    break;

                case "neem oil":
                    imageView.setImageResource(R.drawable.no);
                    translationHelper.translateTextView(title, "Neem Oil");
                    translationHelper.translateTextView(description, "Neem oil is a natural pesticide and insect repellent extracted from neem tree seeds. It disrupts the growth and reproduction of pests such as aphids, whiteflies, and caterpillars. Neem oil is biodegradable and safe for beneficial insects like bees. It also has antifungal properties, protecting crops from diseases like powdery mildew.");
                    translationHelper.translateTextView(keyIngredients, "Azadirachtin, Nimbin, Salannin.");
                    translationHelper.translateTextView(source, "Natural (neem tree seeds).");
                    break;

                case "pyr":
                    imageView.setImageResource(R.drawable.pyth);
                    translationHelper.translateTextView(title, "Pyrethrin");
                    translationHelper.translateTextView(description, "Pyrethrin is a natural insecticide derived from chrysanthemum flowers. It attacks the nervous system of insects, causing paralysis and death. It is effective against mosquitoes, beetles, flies, and caterpillars. Pyrethrin breaks down quickly in sunlight, making it environmentally friendly. Farmers use pyrethrin-based sprays as part of integrated pest management.");
                    translationHelper.translateTextView(keyIngredients, "Pyrethrin I and II compounds.");
                    translationHelper.translateTextView(source, "Natural (chrysanthemum flowers).");
                    break;

                case "dia":
                    imageView.setImageResource(R.drawable.dis);
                    translationHelper.translateTextView(title, "Diatomaceous Earth");
                    translationHelper.translateTextView(description, "Diatomaceous earth is a natural powder made from fossilized remains of microscopic algae called diatoms. It works as a mechanical insecticide by dehydrating pests such as ants, beetles, and mites. Farmers spread it around crops, storage areas, or animal sheds for pest control. It also improves soil aeration when mixed with soil.");
                    translationHelper.translateTextView(keyIngredients, "Silica, alumina, trace minerals.");
                    translationHelper.translateTextView(source, "Natural (fossilized diatoms).");
                    break;

                case "urea":
                    imageView.setImageResource(R.drawable.urea);
                    translationHelper.translateTextView(title, "Urea");
                    translationHelper.translateTextView(description, "Urea is a widely used synthetic nitrogen fertilizer that provides plants with a concentrated source of nitrogen, essential for leaf growth and photosynthesis. It dissolves easily in water, making it quick-acting. Farmers often apply urea during crop growth stages to boost productivity.");
                    translationHelper.translateTextView(keyIngredients, "Nitrogen (46%).");
                    translationHelper.translateTextView(source, "Synthetic (from ammonia and carbon dioxide).");
                    break;

                case "dap":
                    imageView.setImageResource(R.drawable.dap);
                    translationHelper.translateTextView(title, "DAP (Diammonium Phosphate)");
                    translationHelper.translateTextView(description, "DAP is a popular fertilizer containing both nitrogen and phosphorus. It is highly soluble in water, making nutrients available quickly to crops. Farmers use DAP during sowing to promote root development and early plant growth. Its balanced nutrient supply enhances yield and strengthens plants against stress.");
                    translationHelper.translateTextView(keyIngredients, "Nitrogen (18%), Phosphorus (46%).");
                    translationHelper.translateTextView(source, "Synthetic (from ammonia and phosphoric acid).");
                    break;

                case "mop":
                    imageView.setImageResource(R.drawable.mop);
                    translationHelper.translateTextView(title, "MOP (Muriate of Potash)");
                    translationHelper.translateTextView(description, "MOP is the most common potassium fertilizer used worldwide. Potassium strengthens plant stems, improves drought resistance, and enhances crop quality. Farmers apply MOP to crops like sugarcane, potato, banana, and vegetables. It also improves storage quality of produce.");
                    translationHelper.translateTextView(keyIngredients, "Potassium chloride (60% K₂O).");
                    translationHelper.translateTextView(source, "Mineral (mined from potash deposits).");
                    break;

                case "car":
                    imageView.setImageResource(R.drawable.carb);
                    translationHelper.translateTextView(title, "Carbaryl");
                    translationHelper.translateTextView(description, "Carbaryl is a synthetic insecticide belonging to the carbamate group. It is effective against chewing and sucking insects such as caterpillars, beetles, and aphids. Farmers use it to protect fruits, vegetables, and field crops. Carbaryl acts on the insect's nervous system, leading to paralysis.");
                    translationHelper.translateTextView(keyIngredients, "Carbaryl (1-naphthyl methylcarbamate).");
                    translationHelper.translateTextView(source, "Synthetic (chemically produced).");
                    break;

                case "chlor":
                    imageView.setImageResource(R.drawable.chlor);
                    translationHelper.translateTextView(title, "Chlorpyrifos");
                    translationHelper.translateTextView(description, "Chlorpyrifos is an organophosphate insecticide used to control soil-borne and foliar pests like termites, cutworms, and borers. It works by disrupting the nervous system of insects. Farmers apply it on cotton, rice, maize, and vegetables. It is effective but should be used safely.");
                    translationHelper.translateTextView(keyIngredients, "Chlorpyrifos compound.");
                    translationHelper.translateTextView(source, "Synthetic (chemical).");
                    break;

                case "gly":
                    imageView.setImageResource(R.drawable.gly);
                    translationHelper.translateTextView(title, "Glyphosate");
                    translationHelper.translateTextView(description, "Glyphosate is a broad-spectrum systemic herbicide used to control weeds. It blocks a plant enzyme essential for growth, leading to weed death. Farmers use it for land preparation before sowing crops like maize, soybean, and cotton. Overuse can cause herbicide resistance in weeds.");
                    translationHelper.translateTextView(keyIngredients, "Glyphosate isopropylamine salt.");
                    translationHelper.translateTextView(source, "Synthetic (chemical).");
                    break;

                default:
                    translationHelper.translateTextView(title, "No Data Available");
                    translationHelper.translateTextView(description, "Sorry, information for this item is not available.");
                    translationHelper.translateTextView(keyIngredients, "-");
                    translationHelper.translateTextView(source, "-");
                    break;
            }
        });

        // Adjust padding for system bars
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
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }



    }


    private void speakDescription(String englishText) {
        if (englishText == null || englishText.isEmpty()) return;

        String selectedLang = translationHelper.getSelectedLanguage(); // "ta" for Tamil, "en" for English

        if ("ta".equals(selectedLang)) {
            // Translate to Tamil first
            translationHelper.translateText(englishText, translatedText -> {
                textToSpeech.setLanguage(new Locale("ta", "IN"));
                textToSpeech.speak(translatedText, TextToSpeech.QUEUE_FLUSH, null, "tts_ta");
            });
        } else {
            // Speak in English
            textToSpeech.setLanguage(Locale.ENGLISH);
            textToSpeech.speak(englishText, TextToSpeech.QUEUE_FLUSH, null, "tts_en");
        }
    }



}
