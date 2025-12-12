package com.example.agrimitra;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.view.View;
import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.widget.ImageView;
import android.widget.TextView;

public class PestDiseaseResult extends AppCompatActivity {

    private TranslationHelper translationHelper;
    private TextView tDiseaseName, tDescription, tCauses, tSymptoms,
            tPrecautions, tPesticides, tCultural, tCostLoss, tHelpline;

    ImageView imageView;
    CardView callButton;
    public String helplineNumber = "8103222607";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_crop_disease_result);

        translationHelper = new TranslationHelper(this);

        imageView = findViewById(R.id.imgDisease);

        tDiseaseName = findViewById(R.id.tDiseaseName);
        tDescription = findViewById(R.id.tDescription);
        tCauses = findViewById(R.id.tCauses);
        tSymptoms = findViewById(R.id.tSymptoms);
        tPrecautions = findViewById(R.id.tPrecautions);
        tPesticides = findViewById(R.id.tPesticides);
        tCultural = findViewById(R.id.tCultural);
        tCostLoss = findViewById(R.id.tCostLoss);
        tHelpline = findViewById(R.id.tHelpline);

        callButton = findViewById(R.id.callButton);
        callButton.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_DIAL);
            intent.setData(Uri.parse("tel:" + helplineNumber));
            startActivity(intent);
        });



        CardView button1=findViewById(R.id.button1);
        CardView button2=findViewById(R.id.button2);
        CardView button3=findViewById(R.id.button3);
        CardView button4=findViewById(R.id.button4);


        button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Generic-cvt4-Tagmycin-Streptomycin-Sulphate/dp/B0BZLBPBZS?utm_source=chatgpt.com"));
        button2.setOnClickListener(v -> openUrl("http://amazon.in/Syngenta-Amistar-Fungicide-100ml-Pack/dp/B09R34T27Q?utm_source=chatgpt.com"));
        button3.setOnClickListener(v -> openUrl("https://www.amazon.in/biosciences-Organic-Pesticide-Trichoderma-Pseudomonas/dp/B07VYZ6M22?utm_source=chatgpt.com"));
        button4.setOnClickListener(v -> openUrl("https://www.amazon.in/500GMS-HEXACONZOLE4-Contact-Fungicide-Systemic/dp/B07D2H917W?utm_source=chatgpt.com"));





        String imagePath = getIntent().getStringExtra("image_path");
        if (imagePath != null) {
            imageView.setImageURI(Uri.parse(imagePath));
        }

        String predictedClass = getIntent().getStringExtra("predicted_pest");
        if (predictedClass != null) {
            showPestInfo(predictedClass);
        }



        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void showPestInfo(String pest) {
        switch (pest) {

            // ---------- Rice Pests ----------
            case "asiatic rice borer":
                setInfo("Asiatic Rice Borer",
                        "A pest that bores into rice stems affecting growth.",
                        "Larvae bore into stem and panicle.",
                        "Dead hearts, whiteheads, reduced grain filling.",
                        "Regular field monitoring, remove affected plants.",
                        "Use chlorantraniliprole or neem-based insecticides.",
                        "Maintain healthy crop, avoid dense planting.",
                        "Yield loss can reach 20-40%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "brown plant hopper":
                setInfo("Brown Plant Hopper",
                        "Sap-sucking pest causing hopperburn in rice.",
                        "Feeds on plant sap, transmits viruses.",
                        "Yellowing, stunted growth, hopperburn symptoms.",
                        "Water management, remove infected plants.",
                        "Use Imidacloprid, Thiamethoxam.",
                        "Crop rotation, maintain field hygiene.",
                        "Yield loss 15-60%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "paddy stem maggot":
                setInfo("Paddy Stem Maggot",
                        "Larvae attack young seedlings causing stunted growth.",
                        "Eggs laid in leaf sheaths, larvae bore into stems.",
                        "Wilting, stunted plants, blackened stems.",
                        "Maintain water level, remove infested seedlings.",
                        "Apply Chlorpyrifos or Quinalphos.",
                        "Avoid excessive nitrogen.",
                        "Moderate yield reduction.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice gall midge":
                setInfo("Rice Gall Midge",
                        "Small insect causing silver shoots in rice.",
                        "Larvae feed inside shoots forming galls.",
                        "Silver shoots, no panicle development.",
                        "Remove infested seedlings, early planting.",
                        "Apply Carbofuran granules in nursery.",
                        "Maintain field sanitation.",
                        "Yield loss can reach 20-50%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice leaf caterpillar":
                setInfo("Rice Leaf Caterpillar",
                        "Caterpillars feed on leaf tissues.",
                        "Larvae feed on leaf blades.",
                        "Leaf defoliation, holes in leaves.",
                        "Regular monitoring and manual removal.",
                        "Apply Bacillus thuringiensis or synthetic insecticides.",
                        "Maintain proper spacing.",
                        "Minor to moderate yield loss.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice leaf roller":
                setInfo("Rice Leaf Roller",
                        "Rolls leaves and feeds inside causing damage.",
                        "Larvae roll leaves and eat internal tissues.",
                        "Rolled leaves, reduced photosynthesis.",
                        "Destroy rolled leaves, field monitoring.",
                        "Use insecticides like Cartap Hydrochloride.",
                        "Maintain proper irrigation and field hygiene.",
                        "Yield loss 10-30%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice leafhopper":
                setInfo("Rice Leafhopper",
                        "Sap-sucking pest causing hopperburn and transmitting viruses.",
                        "Feeds on phloem, transmits rice tungro virus.",
                        "Yellowing, stunted growth, hopperburn.",
                        "Field sanitation, remove infected plants.",
                        "Imidacloprid or Thiamethoxam sprays.",
                        "Maintain optimal water management.",
                        "Yield loss 15-50%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice shell pest":
                setInfo("Rice Shell Pest",
                        "Damages developing grains causing empty shells.",
                        "Larvae feed inside rice grains.",
                        "Hollow grains, reduced grain quality.",
                        "Proper drying and storage.",
                        "Use insecticides like Chlorpyrifos.",
                        "Maintain field hygiene.",
                        "Moderate yield loss.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "rice water weevil":
                setInfo("Rice Water Weevil",
                        "Larvae feed on roots causing stunted growth.",
                        "Adults feed on leaves, larvae on roots.",
                        "Reduced tillering, stunted seedlings.",
                        "Drain water periodically to manage adults.",
                        "Apply systemic insecticides.",
                        "Maintain proper water management.",
                        "Yield loss 10-25%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "small brown plant hopper":
                setInfo("Small Brown Plant Hopper",
                        "Sap-sucking pest causing hopperburn and transmitting viruses.",
                        "Feeds on sap, transmits rice tungro virus.",
                        "Yellowing, stunted growth, hopperburn.",
                        "Maintain field hygiene, remove infected plants.",
                        "Use Imidacloprid or Thiamethoxam.",
                        "Crop rotation recommended.",
                        "Yield loss 10-50%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "white backed plant hopper":
                setInfo("White Backed Plant Hopper",
                        "Causes hopperburn, sucks sap, transmits viral diseases.",
                        "Feeds on plant sap, vectors viruses.",
                        "Yellowing, drying, stunted growth.",
                        "Remove infected seedlings, maintain proper water.",
                        "Apply systemic insecticides.",
                        "Field sanitation.",
                        "Yield loss 20-60%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "yellow rice borer":
                setInfo("Yellow Rice Borer",
                        "Bores into stem affecting grain filling.",
                        "Larvae bore into stem and panicle.",
                        "Dead hearts, whiteheads.",
                        "Monitor fields, remove affected plants.",
                        "Apply Chlorantraniliprole or Neem oil.",
                        "Crop rotation and healthy crop practices.",
                        "Yield loss 15-40%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            // ---------- Cotton Pests ----------
            case "bollworm":
                setInfo("Cotton Bollworm",
                        "Larvae attack bolls causing damage to cotton fiber.",
                        "Feeds on flowers and young bolls.",
                        "Holes in bolls, premature opening, reduced quality.",
                        "Field monitoring, remove affected bolls.",
                        "Use Spinosad, Chlorantraniliprole.",
                        "Intercrop, maintain proper spacing.",
                        "Yield loss 20-50%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "stem borer":
                setInfo("Cotton Stem Borer",
                        "Larvae bore into stem affecting plant growth.",
                        "Bores into main stem and branches.",
                        "Wilting, dead hearts, broken stems.",
                        "Remove affected plants, early sowing.",
                        "Apply Carbofuran, Chlorpyrifos.",
                        "Maintain crop hygiene.",
                        "Yield loss 15-40%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "whitefly":
                setInfo("Cotton Whitefly",
                        "Sap-sucking pest transmitting viral diseases.",
                        "Feeds on phloem, secretes honeydew.",
                        "Yellowing, leaf curling, reduced growth.",
                        "Monitor population, remove weeds.",
                        "Use Imidacloprid or Neem oil.",
                        "Maintain proper spacing.",
                        "Yield loss 10-40%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            // ---------- Wheat Pests ----------
            case "sawfly":
                setInfo("Wheat Sawfly",
                        "Larvae feed on stems causing lodging.",
                        "Feed inside stems weakening plant.",
                        "Bent stems, poor grain filling.",
                        "Destroy infested stubbles.",
                        "Apply insecticides at early stage.",
                        "Maintain field hygiene.",
                        "Yield loss 10-30%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "wheat blossom midge":
                setInfo("Wheat Blossom Midge",
                        "Larvae feed on developing flowers affecting grain formation.",
                        "Adults lay eggs in wheat flowers.",
                        "Empty grains, reduced grain number.",
                        "Early sowing, resistant varieties.",
                        "Apply insecticides at heading stage.",
                        "Field sanitation.",
                        "Yield loss 15-50%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "wheat phloeothrips":
                setInfo("Wheat Phloeothrips",
                        "Feeds on leaves causing discoloration.",
                        "Sap-sucking, feeds on young leaves.",
                        "Yellowing, curling of leaves.",
                        "Monitor population, remove affected plants.",
                        "Use insecticides like Thiamethoxam.",
                        "Maintain crop hygiene.",
                        "Yield loss minor to moderate.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            case "wheat sawfly":
                setInfo("Wheat Sawfly",
                        "Larvae bore into stems causing lodging.",
                        "Feed inside stems weakening plant.",
                        "Bent stems, poor grain filling.",
                        "Destroy infested stubbles.",
                        "Apply insecticides at early stage.",
                        "Maintain field hygiene.",
                        "Yield loss 10-30%.",
                        "Call Kisan Helpline 1800-180-1551");
                break;

            // ---------- Default ----------
            default:
                setInfo("Unknown Pest", "No data available.", "-", "-", "-", "-", "-", "-", "-");
                break;
        }
    }

    private void setInfo(String name, String desc, String causes, String symptoms,
                         String precautions, String pesticides, String cultural,
                         String costLoss, String helpline) {

        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(tDiseaseName, name);
            translationHelper.translateTextView(tDescription, desc);
            translationHelper.translateTextView(tCauses, causes);
            translationHelper.translateTextView(tSymptoms, symptoms);
            translationHelper.translateTextView(tPrecautions, precautions);
            translationHelper.translateTextView(tPesticides, pesticides);
            translationHelper.translateTextView(tCultural, cultural);
            translationHelper.translateTextView(tCostLoss, costLoss);
            translationHelper.translateTextView(tHelpline, helpline);
        });
    }


    private void openUrl(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
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
