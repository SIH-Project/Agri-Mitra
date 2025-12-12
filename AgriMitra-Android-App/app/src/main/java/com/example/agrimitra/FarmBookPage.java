package com.example.agrimitra;

import android.os.Bundle;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.speech.tts.TextToSpeech;
import java.util.Locale;


public class FarmBookPage extends AppCompatActivity {

    private TranslationHelper translationHelper;

    private TextToSpeech textToSpeech;

    String text;


    TextView t2,t4,t6,t8,t10,t12,t14,t16,t18,t20,t22,submit_text,maintitle;







    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_farm_book_page);

        translationHelper = new TranslationHelper(this);

        Spinner spinner = findViewById(R.id.spinnerPestType);

        String[] crops = {"அரிசி", "ଗହମ", "Corn", "Potato", "Tomato"};


        makeChanges("அரிசி");

        maintitle=findViewById(R.id.hello);







        translationHelper.initializeTranslation(() -> {
            ArrayAdapter<String> adapter = new ArrayAdapter<>(
                    this,
                    R.layout.spinner_item,
                    crops
            );
            adapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinner.setAdapter(adapter);
        });



        CardView submit=findViewById(R.id.btnSubmit);

        t2 = findViewById(R.id.t2);
        submit_text = findViewById(R.id.abc);
        t4 = findViewById(R.id.t4);
        t6 = findViewById(R.id.t6);
         t8 = findViewById(R.id.t8);
         t10 = findViewById(R.id.t10);
         t12 = findViewById(R.id.t12);
         t14 = findViewById(R.id.t14);
         t16 = findViewById(R.id.t16);
         t18 = findViewById(R.id.t18);
         t20 = findViewById(R.id.t20);
         t22 = findViewById(R.id.t22);


        TextView t7 = findViewById(R.id.t7);
        TextView t9 = findViewById(R.id.t9);
        TextView t11 = findViewById(R.id.t11);
        TextView t15 = findViewById(R.id.t15);
        TextView t19 = findViewById(R.id.t19);
        TextView t21 = findViewById(R.id.t21);
        TextView t23 = findViewById(R.id.t23);




        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(submit_text,
                    "Submit");

            if (t7 != null) translationHelper.translateTextView(t7,
                    "1. Choose high-yield, disease-resistant, and locally adapted varieties.\n\n" +
                            "2. Consider growth duration (short, medium, or long-duration varieties) based on local climate and market demand"
            );
            if (t9 != null) translationHelper.translateTextView(t9,
                    "1. Use certified seeds with high germination rates.\n\n" +
                            "2. Soak seeds in water for 24 hours, then incubate in a wet cloth for 24-48 hours until sprouting."
            );
            if (t11 != null) translationHelper.translateTextView(t11,
                    "1. Transplanting: Seedlings are uprooted from the nursery and transplanted into the main field.\n\n" +
                            "2. Direct Seeding: Seeds are sown directly into the field either dry or pre-germinated."
            );
            if (t15 != null) translationHelper.translateTextView(t15,
                    "Essential nutrients: Nitrogen (N), Phosphorus (P), Potassium (K).\n\n" +
                            "Basal application: Apply phosphorus and potassium during land preparation."
            );
            if (t19 != null) translationHelper.translateTextView(t19,
                    "Harvest when grains are mature (about 20-25% moisture content in grains).\n\n" +
                            "Use sickles or mechanical harvesters."
            );
            if (t21 != null) translationHelper.translateTextView(t21,
                    "Dry grains immediately after harvest to reduce moisture content to 12-14%.\n\n" +
                            "Proper drying prevents fungal growth and preserves grain quality."
            );
            if (t23 != null) translationHelper.translateTextView(t23,
                    "Practice crop rotation with legumes or other crops to improve soil health.\n\n" +
                            "Use organic fertilizers and bio-pesticides."
            );
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
            text=spinner.getSelectedItem().toString();
            speakCropDescription(text);
        });







        translationHelper.initializeTranslation(() -> {
            if (t2 != null) translationHelper.translateTextView(t2, "Description");
            if (t4 != null) translationHelper.translateTextView(t4, "Land Preparation");
            if (t6 != null) translationHelper.translateTextView(t6, "Selecting Variety");
            if (t8 != null) translationHelper.translateTextView(t8, "Seed Selection");
            if (t10 != null) translationHelper.translateTextView(t10, "Planting Methods");
            if (t12 != null) translationHelper.translateTextView(t12, "Water Management");
            if (t14 != null) translationHelper.translateTextView(t14, "Fertilizer Application");
            if (t16 != null) translationHelper.translateTextView(t16, "Pest and Disease Management");
            if (t18 != null) translationHelper.translateTextView(t18, "Harvesting");
            if (t20 != null) translationHelper.translateTextView(t20, "Post-Harvest Handling");
            if (t22 != null) translationHelper.translateTextView(t22, "Sustainable Practices");
            if (maintitle != null) translationHelper.translateTextView(maintitle, "Farm Guide");


        });


















        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                text=spinner.getSelectedItem().toString().toLowerCase();
                makeChanges(text);
            }
        });








        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    public void makeChanges(String crop)
    {

        ImageView imageView=findViewById(R.id.img1);
        TextView title=findViewById(R.id.t1);
        TextView description=findViewById(R.id.t3);
        TextView landPreparation=findViewById(R.id.t5);
        TextView plantingMethod=findViewById(R.id.t11);
        TextView waterManagement=findViewById(R.id.t13);
        TextView pest=findViewById(R.id.t17);


        if (crop.equals("ଗହମ")) {
            imageView.setImageResource(R.drawable.wheat);
//            title.setText("Wheat");
//            description.setText("Wheat is a staple cereal crop grown worldwide for its grain, which is milled into flour for bread, pasta, and other foods. It thrives in temperate climates with moderate rainfall and well-drained soil. Wheat is rich in carbohydrates and protein, making it vital for human nutrition.");
//            landPreparation.setText("Plough the land to a fine tilth, remove weeds, and level the field. Add organic manure and fertilizers based on soil testing.");
//            plantingMethod.setText("Sow seeds using a seed drill for uniform spacing or broadcast manually, then lightly cover with soil.");
//            waterManagement.setText("Requires moderate irrigation, especially at tillering, flowering, and grain filling stages.");
//            pest.setText("Use resistant varieties, timely fungicide sprays, and crop rotation to manage pests like aphids and rust.");

            title.setText("ଗହମ");
            description.setText("ଧାନ୍ୟ ହିସାବରେ ଗହମ୍ ଏକ ପ୍ରମୁଖ ଆହାର ଫସଲ୍ ଯାହା ବିଶ୍ୱବ୍ୟାପୀ ଅର୍ଜିତ ହୁଏ, ଏହାର ଦାନା ପିଠା ତଳା, ପାସ୍ତା ଓ ଅନ୍ୟାନ୍ୟ ଖାଦ୍ୟ ପଦାର୍ଥ ପାଇଁ ଚାକୁଆରେ ପିସାଯାଏ। ଗହମ୍ ସାଧାରଣ ପାଣିପାଗ ଓ ଭଲ ଭାବରେ ନିକାସ ହେବା ଦେୟ ମାଟି ଥିବା ସାଧାରଣ ଜଳବାୟୁରେ ଭଲ ଭାବରେ ବଢେ। ଗହମ୍ କାର୍ବୋହାଇଡ୍ରେଟ୍ ଏବଂ ପ୍ରୋଟିନ୍ରେ ଧନ୍ୟ, ଯାହାକୁ ମାନବ ପୋଷଣର ପାଇଁ ଅତିଆବଶ୍ୟକ କରେ।");
            landPreparation.setText("ମାଟିକୁ ସୁଖାମୌଳି ଧରଣରେ ଖଣ୍ଡିବା, ଅନାବଶ୍ୟକ ବୁଟା ଉଠାଇବା ଏବଂ କ୍ଷେତ୍ରକୁ ସମତଳ କରିବା। ମାଟିର ପରୀକ୍ଷା ଅନୁଯାୟୀ ସାଂଗରିକ ସୁରକ୍ଷା ଓ ରସାୟନିକ ସର୍ବାଧାନକ ଯୋଗ କରିବେ।");
                plantingMethod.setText("ବିଜ ମେଶିନ୍ ବ୍ୟବହାର କରି ସମାନ ଦୂରତାରେ ବୀଜ ପୋଆନ୍ତୁ କିମ୍ବା ହସ୍ତ ମାନ୍ୟ କରି ବୀଜ ପୋଇବେ, ତାପରେ ସାମାନ୍ୟ ମାଟିରେ ଢକିଦିଅନ୍ତୁ।");
            waterManagement.setText("ମଧ୍ୟମ ପରିମାଣର ସିଚାଇ ଆବଶ୍ୟକ, ବିଶେଷତ: ଟିଲରିଙ୍ଗ, ଫୁଲ ଫାଲିବା ଏବଂ ଧାନ ପୂରଣ ଅବସ୍ଥାରେ।");
            pest.setText("ପୋକମାକୁ ଓ ଧୁଳାଇଓଁ ପରି କୀଟକୁ ନିୟନ୍ତ୍ରଣ କରିବା ପାଇଁ ପ୍ରତିରୋଧକ ପ୍ରଜାତି, ସମୟୋପଯୁକ୍ତ ଫଙ୍ଗିସାଇଡ୍ ଛିଟା ଏବଂ ଫସଲ ଘୁର୍ଣ୍ଣନୀ ବ୍ୟବହାର କରନ୍ତୁ।");
            t2.setText("ବର୍ଣ୍ଣନା");
            t4.setText("ଭୂମି ପ୍ରସ୍ତୁତି");
            t6.setText("ବିବିଧତା ଚୟନ କରିବା");
            t8.setText("ବୀଜ ବଛନା");
            t10.setText("ଚାଷ ପদ্ধତି");
            t12.setText("ଜଳ ପରିଚାଳନା");
            t14.setText("ଉର୍ବରକ ପ୍ରୟୋଗ");
            t16.setText("ପୋକମାକୁଡ଼ି ଓ ରୋଗ ପରିଚାଳନା");
            t18.setText("କାଟଣା");
            t20.setText("ହର୍ଭେଷ୍ଟ ପରେ ହେବା ଥିବା ପରିଚର୍ୟା");
            t22.setText("ଟିକାଉ ପ୍ରଥାମାନେ");
            submit_text.setText("ଦାଖଲ କରନ୍ତୁ");
            maintitle.setText("ଖେତି ମାର୍ଗଦର୍ଶନ");





        } else if (crop.equals("அரிசி")) {
            imageView.setImageResource(R.drawable.rice);
            title.setText("Rice");
            description.setText("Rice is a major food crop grown primarily in flooded fields called paddies. It requires warm temperatures and abundant water. Rice is a carbohydrate-rich staple feeding over half the world’s population, especially in Asia.");
            landPreparation.setText("Plough the field and puddle the soil by flooding and mixing to create a soft, muddy bed for transplantation.");
            plantingMethod.setText("Raise seedlings in nurseries; transplant 20-30 days old seedlings into flooded fields.");
            waterManagement.setText("Maintain continuous shallow flooding until grain filling, then gradually drain water before harvest.");
            pest.setText("Use integrated pest management (IPM) including resistant varieties, natural predators, and safe insecticides for pests like stem borers and leafhoppers.");

        } else if (crop.equals("corn")) {
            imageView.setImageResource(R.drawable.corn);
            title.setText("Corn");
            description.setText("Corn is a versatile cereal grown for food, fodder, and industrial uses. It grows well in warm climates with well-drained fertile soils. Corn is rich in carbohydrates and widely used for human consumption and livestock feed.");
            landPreparation.setText("Plough and harrow the soil to a fine tilth, ensuring good drainage and aeration.");
            plantingMethod.setText("Sow seeds directly in rows using seed drills or manual planting, spacing 60-90 cm apart.");
            waterManagement.setText("Requires regular watering, especially during tasseling and silking stages; avoid waterlogging.");
            pest.setText("Use resistant hybrids, crop rotation, and pesticides for pests like corn borers and armyworms.");

        } else if (crop.equals("potato")) {
            imageView.setImageResource(R.drawable.potato);
            title.setText("Potato");
            description.setText("Potato is a tuber crop grown for its nutritious edible tubers rich in carbohydrates, vitamins, and minerals. It grows best in cool climates and well-drained, loose soils.");
            landPreparation.setText("Plough and add organic matter; create ridges or mounds to improve drainage and tuber development.");
            plantingMethod.setText("Plant seed tubers or cut pieces with eyes spaced 20-30 cm apart on ridges.");
            waterManagement.setText("Keep soil moist but not waterlogged; frequent watering during tuber formation is critical.");
            pest.setText("Control pests like potato beetles and aphids using insecticides and crop rotation.");

        } else if (crop.equals("tomato")) {
            imageView.setImageResource(R.drawable.tomato);
            title.setText("Tomato");
            description.setText("Tomato is a widely cultivated vegetable crop used fresh or processed. It requires warm temperatures, plenty of sunlight, and well-drained fertile soil.");
            landPreparation.setText("Prepare well-tilled, fertile soil enriched with organic matter. Beds or rows should be leveled.");
            plantingMethod.setText("Start seeds in nurseries; transplant seedlings to the field after 4-6 weeks with spacing of 45-60 cm.");
            waterManagement.setText("Water regularly to keep soil moist; avoid wetting leaves to prevent fungal diseases.");
            pest.setText("Manage pests like aphids and tomato fruit worms using insecticides, traps, and resistant varieties.");
        }



        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(title, title.getText().toString());
            translationHelper.translateTextView(description, description.getText().toString());
            translationHelper.translateTextView(landPreparation, landPreparation.getText().toString());
            translationHelper.translateTextView(plantingMethod, plantingMethod.getText().toString());
            translationHelper.translateTextView(waterManagement, waterManagement.getText().toString());
            translationHelper.translateTextView(pest, pest.getText().toString());




        });










    }



    private void speakCropDescription(String crop) {
        String message = "No Description";

        switch (crop) {
            case "rice":
                message = "Rice is a major food crop grown primarily in flooded fields called paddies. It requires warm temperatures and abundant water.";
                break;
            case "wheat":
                message = "Wheat is a staple cereal crop grown worldwide for its grain, which is milled into flour for bread, pasta, and other foods.";
                break;
            case "corn":
                message = "Corn is a versatile cereal grown for food, fodder, and industrial uses. It grows well in warm climates with well-drained fertile soils.";
                break;
            case "potato":
                message = "Potato is a tuber crop grown for its nutritious edible tubers rich in carbohydrates, vitamins, and minerals.";
                break;
            case "tomato":
                message = "Tomato is a widely cultivated vegetable crop used fresh or processed. It requires warm temperatures and well-drained fertile soil.";
                break;
            default:
                message = "Rice is a major food crop grown primarily in flooded fields called paddies.";
                break;
        }

        // Handle multilingual TTS
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
}