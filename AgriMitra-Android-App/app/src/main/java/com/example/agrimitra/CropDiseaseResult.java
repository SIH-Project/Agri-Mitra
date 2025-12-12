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
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import android.speech.tts.TextToSpeech;
import java.util.Locale;


public class CropDiseaseResult extends AppCompatActivity {

    private TranslationHelper translationHelper;
    private TextView tDiseaseName, tDescription, tCauses, tSymptoms,
            tPrecautions, tPesticides, tCultural, tCostLoss, tHelpline;


    private TextToSpeech textToSpeech;




    TextView lblDescription,lblCauses,lblSymptoms,lblPrecautions,lblPesticides,card_text2,card_text3,card_text4,card_text5,b1,b2,b3,b4;
    TextView lblCultural,lblCostLoss,lblHelpline,call;

    ImageView imageView;



    CardView callButton;

    public String helplineNumber="8103222607";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_crop_disease_result);

        translationHelper = new TranslationHelper(this);

        imageView=findViewById(R.id.imgDisease);


        lblDescription = findViewById(R.id.lblDescription);
        lblCauses = findViewById(R.id.lblCauses);
        lblSymptoms = findViewById(R.id.lblSymptoms);
        lblPrecautions = findViewById(R.id.lblPrecautions);
        lblPesticides = findViewById(R.id.lblPesticides);

        card_text2 = findViewById(R.id.card_text2);
        card_text3 = findViewById(R.id.card_text3);
        card_text4 = findViewById(R.id.card_text4);
        card_text5 = findViewById(R.id.card_text5);

        b1 = findViewById(R.id.b1);
        b2 = findViewById(R.id.b2);
        b3 = findViewById(R.id.b3);
        b4 = findViewById(R.id.b4);

        lblCultural = findViewById(R.id.lblCultural);
        lblCostLoss = findViewById(R.id.lblCostLoss);
        lblHelpline = findViewById(R.id.lblHelpline);
        call = findViewById(R.id.call);




        CardView button1=findViewById(R.id.button1);
        CardView button2=findViewById(R.id.button2);
        CardView button3=findViewById(R.id.button3);
        CardView button4=findViewById(R.id.button4);


        button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Generic-cvt4-Tagmycin-Streptomycin-Sulphate/dp/B0BZLBPBZS?utm_source=chatgpt.com"));
        button2.setOnClickListener(v -> openUrl("http://amazon.in/Syngenta-Amistar-Fungicide-100ml-Pack/dp/B09R34T27Q?utm_source=chatgpt.com"));
        button3.setOnClickListener(v -> openUrl("https://www.amazon.in/biosciences-Organic-Pesticide-Trichoderma-Pseudomonas/dp/B07VYZ6M22?utm_source=chatgpt.com"));
        button4.setOnClickListener(v -> openUrl("https://www.amazon.in/500GMS-HEXACONZOLE4-Contact-Fungicide-Systemic/dp/B07D2H917W?utm_source=chatgpt.com"));










        tDiseaseName = findViewById(R.id.tDiseaseName);
        tDescription = findViewById(R.id.tDescription);
        tCauses = findViewById(R.id.tCauses);
        tSymptoms = findViewById(R.id.tSymptoms);
        tPrecautions = findViewById(R.id.tPrecautions);
        tPesticides = findViewById(R.id.tPesticides);
        tCultural = findViewById(R.id.tCultural);
        tCostLoss = findViewById(R.id.tCostLoss);
        tHelpline = findViewById(R.id.tHelpline);


        callButton=findViewById(R.id.callButton);


        textToSpeech = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.ENGLISH);
                textToSpeech.setPitch(1.0f);
                textToSpeech.setSpeechRate(1.0f);
            }
        });







        callButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(Intent.ACTION_DIAL);
                intent.setData(Uri.parse("tel:" + helplineNumber));
                startActivity(intent);
            }
        });


        String predictedClass = getIntent().getStringExtra("crop");


        ImageButton explainButton = findViewById(R.id.explain);
        explainButton.setOnClickListener(v -> {
            if (textToSpeech != null) {
                switch (predictedClass) {

                    // ---------- Cotton ----------
                    case "Cotton_Bacterial_Blight":
                        textToSpeech.speak("Cotton Bacterial Blight. A bacterial disease causing angular leaf spots and boll rot. Caused by Xanthomonas campestris pv. malvacearum." +
                                "Angular black spots on leaves, stem cankers, boll rot." +
                                "Use resistant varieties, avoid overhead irrigation." +
                                "Spray Copper oxychloride or Streptomycin formulations." +
                                "Crop rotation, remove crop debris." +
                                "May reduce yield by 10–30%.", TextToSpeech.QUEUE_FLUSH, null, "tts1");
                        break;

                    case "Cotton_Curl_Virus":
                        textToSpeech.speak("Cotton Curl Virus. A viral disease spread by whiteflies." +
                                "Caused by Cotton Leaf Curl Virus transmitted via Bemisia tabaci." +
                                "Upward curling of leaves, vein thickening, stunted growth." +
                                "Control whitefly population." +
                                "Use Imidacloprid or Thiamethoxam as per guidelines." +
                                "Remove alternate host plants." +
                                "Severe infection can cause more than 50% yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts2");
                        break;

                    case "Cotton_Fusarium_Wilt":
                        textToSpeech.speak("Cotton Fusarium Wilt. Fungal disease leading to wilting of cotton plants." +
                                "Caused by Fusarium oxysporum f.sp. vasinfectum." +
                                "Yellowing, wilting, brown vascular tissues." +
                                "Use disease-free seeds and resistant varieties." +
                                "Apply Carbendazim or Triazole fungicides." +
                                "Crop rotation with cereals." +
                                "Can reduce yield up to 40%.", TextToSpeech.QUEUE_FLUSH, null, "tts3");
                        break;

                    case "Cotton_Healthy":
                        textToSpeech.speak("Healthy Cotton Plant. No disease detected." +
                                "Continue regular monitoring and good practices." +
                                "Maintain proper spacing and irrigation." +
                                "No yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts4");
                        break;

                    // ---------- Sugarcane ----------
                    case "Sugarcane_Bacterial_Blight":
                        textToSpeech.speak("Sugarcane Bacterial Blight. Serious bacterial disease in sugarcane." +
                                "Caused by Xanthomonas albilineans." +
                                "Leaf scalding, white streaks, plant death." +
                                "Use resistant varieties." +
                                "Spray Copper compounds as preventive." +
                                "Destroy diseased clumps." +
                                "Can cause up to 30% yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts5");
                        break;

                    case "Sugarcane_Mosaic":
                        textToSpeech.speak("Sugarcane Mosaic Virus. Viral disease affecting sugarcane leaves." +
                                "Caused by Sugarcane mosaic virus." +
                                "Mosaic yellow-green patterns on leaves." +
                                "Plant resistant varieties." +
                                "No direct chemical control; vector management is needed." +
                                "Remove infected plants." +
                                "Losses up to 20–30%.", TextToSpeech.QUEUE_FLUSH, null, "tts6");
                        break;

                    case "Sugarcane_Red_Rot":
                        textToSpeech.speak("Sugarcane Red Rot. One of the most destructive fungal diseases." +
                                "Caused by Colletotrichum falcatum." +
                                "Reddish discoloration inside stalk, foul smell." +
                                "Use resistant varieties." +
                                "Apply Carbendazim or Propiconazole." +
                                "Crop rotation and field sanitation." +
                                "Yield losses up to 70%.", TextToSpeech.QUEUE_FLUSH, null, "tts7");
                        break;

                    case "Sugarcane_Rust":
                        textToSpeech.speak("Sugarcane Rust. Fungal disease forming pustules on leaves." +
                                "Caused by Puccinia melanocephala." +
                                "Rusty brown pustules on lower leaf surfaces." +
                                "Plant resistant cultivars." +
                                "Use fungicides like Mancozeb or Triazoles." +
                                "Remove severely infected leaves." +
                                "Losses 10–40%.", TextToSpeech.QUEUE_FLUSH, null, "tts8");
                        break;

                    case "Sugarcane_Yellow":
                        textToSpeech.speak("Sugarcane Yellow Leaf Disease. Viral disease transmitted by aphids." +
                                "Caused by Sugarcane yellow leaf virus." +
                                "Midrib yellowing progressing to whole leaf." +
                                "Control aphid population." +
                                "No direct cure; manage with insecticides." +
                                "Use virus-free planting material." +
                                "Moderate yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts9");
                        break;

                    case "Sugarcane_Healthy":
                        textToSpeech.speak("Healthy Sugarcane. No disease detected." +
                                "Continue good practices." +
                                "Maintain crop hygiene." +
                                "No yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts10");
                        break;

                    // ---------- Rice ----------
                    case "Rice_Tungro":
                        textToSpeech.speak("Rice Tungro. Viral disease causing stunted growth and yellow-orange discoloration." +
                                "Caused by Rice tungro bacilliform virus and Rice tungro spherical virus transmitted by leafhoppers." +
                                "Yellowing of leaves, stunted plant growth." +
                                "Use resistant varieties and control leafhopper population." +
                                "No direct chemical treatment; manage vector population." +
                                "Remove infected plants." +
                                "May reduce yield by 50–70%.", TextToSpeech.QUEUE_FLUSH, null, "tts11");
                        break;

                    case "Rice_Sheath_Blight":
                        textToSpeech.speak("Rice Sheath Blight. Fungal disease affecting rice sheath and leaves." +
                                "Caused by Rhizoctonia solani." +
                                "Lesions on leaf sheaths, spreading to leaves, reduced tillering." +
                                "Ensure proper spacing and water management." +
                                "Use valid fungicides such as Hexaconazole or Propiconazole." +
                                "Crop rotation with non-host crops." +
                                "Yield loss may reach 20-50%.", TextToSpeech.QUEUE_FLUSH, null, "tts12");
                        break;

                    case "Rice_Leaf_Scald":
                        textToSpeech.speak("Rice Leaf Scald. Fungal disease causing white to grey lesions on leaves." +
                                "Caused by Microdochium oryzae." +
                                "White-grey streaks on leaves with yellow borders." +
                                "Maintain proper water and nutrient management." +
                                "Apply appropriate fungicides." +
                                "Remove affected plant debris." +
                                "May reduce yield moderately.", TextToSpeech.QUEUE_FLUSH, null, "tts13");
                        break;

                    case "Rice_Brown_Spot":
                        textToSpeech.speak("Rice Brown Spot. Fungal disease causing brown lesions on leaves and grains." +
                                "Caused by Bipolaris oryzae." +
                                "Brown spots on leaves, can reduce grain quality." +
                                "Ensure balanced fertilization, proper spacing." +
                                "Apply recommended fungicides." +
                                "Remove infected residues." +
                                "Moderate yield reduction.", TextToSpeech.QUEUE_FLUSH, null, "tts14");
                        break;

                    case "Rice_Blast":
                        textToSpeech.speak("Rice Blast. Serious fungal disease affecting leaves, nodes, and panicles." +
                                "Caused by Magnaporthe oryzae." +
                                "Diamond-shaped lesions on leaves, neck rot, grain sterility." +
                                "Use resistant varieties and avoid excessive nitrogen." +
                                "Fungicides like Tricyclazole may be used." +
                                "Crop rotation and remove infected debris." +
                                "Yield loss can be 50-70%.", TextToSpeech.QUEUE_FLUSH, null, "tts15");
                        break;

                    case "Rice_Bacterial_Leaf_Blight":
                        textToSpeech.speak("Rice Bacterial Leaf Blight. Bacterial disease causing yellowing and drying of leaf margins." +
                                "Caused by Xanthomonas oryzae pv. oryzae." +
                                "Yellowing of leaf edges, wilting, reduced tillers." +
                                "Use resistant varieties, avoid high nitrogen application." +
                                "Apply Copper-based bactericides." +
                                "Remove infected plant debris." +
                                "Yield loss may reach 20-50%.", TextToSpeech.QUEUE_FLUSH, null, "tts16");
                        break;

                    case "Rice_Healthy":
                        textToSpeech.speak("Healthy Rice Plant. No disease detected." +
                                "Continue good agronomic practices." +
                                "Maintain proper irrigation and fertilization." +
                                "No yield loss.", TextToSpeech.QUEUE_FLUSH, null, "tts17");
                        break;

                    // ---------- Default ----------
                    default:
                        textToSpeech.speak("No information available for this disease.", TextToSpeech.QUEUE_FLUSH, null, "ttsDefault");
                        break;
                }
            }
        });



        if (predictedClass != null) {
            showDiseaseInfo(predictedClass);
        }

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void showDiseaseInfo(String disease) {
        switch (disease) {

            // ---------- Cotton ----------
            case "Cotton_Bacterial_Blight":
                imageView.setImageResource(R.drawable.cotton_blight);
                setInfo(
                        "Cotton Bacterial Blight",
                        "A bacterial disease causing angular leaf spots and boll rot.",
                        "Caused by Xanthomonas campestris pv. malvacearum.",
                        "Angular black spots on leaves, stem cankers, boll rot.",
                        "Use resistant varieties, avoid overhead irrigation.",
                        "Spray Copper oxychloride or Streptomycin formulations.",
                        "Crop rotation, remove crop debris.",
                        "May reduce yield by 10–30%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Cotton_Curl_Virus":
                imageView.setImageResource(R.drawable.cotton_curl);
                setInfo(
                        "Cotton Curl Virus",
                        "A viral disease spread by whiteflies.",
                        "Caused by Cotton Leaf Curl Virus transmitted via Bemisia tabaci.",
                        "Upward curling of leaves, vein thickening, stunted growth.",
                        "Control whitefly population.",
                        "Use Imidacloprid, Thiamethoxam (follow dose guidelines).",
                        "Remove alternate host plants.",
                        "Severe infection can cause >50% yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Cotton_Fusarium_Wilt":
                imageView.setImageResource(R.drawable.cotton_fusari);
                setInfo(
                        "Cotton Fusarium Wilt",
                        "Fungal disease leading to wilting of cotton plants.",
                        "Caused by Fusarium oxysporum f.sp. vasinfectum.",
                        "Yellowing, wilting, brown vascular tissues.",
                        "Use disease-free seeds, resistant varieties.",
                        "Apply Carbendazim or Triazole fungicides.",
                        "Crop rotation with cereals.",
                        "Can reduce yield up to 40%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Cotton_Healthy":
                imageView.setImageResource(R.drawable.cotton_healthy);
                setInfo(
                        "Healthy Cotton Plant",
                        "No disease detected.",
                        "-",
                        "-",
                        "Continue regular monitoring and good practices.",
                        "-",
                        "Maintain proper spacing and irrigation.",
                        "No yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            // ---------- Sugarcane ----------
            case "Sugarcane_Bacterial_Blight":
                imageView.setImageResource(R.drawable.sugarcane_blight);
                setInfo(
                        "Sugarcane Bacterial Blight",
                        "Serious bacterial disease in sugarcane.",
                        "Caused by Xanthomonas albilineans.",
                        "Leaf scalding, white streaks, plant death.",
                        "Use resistant varieties.",
                        "Spray Copper compounds as preventive.",
                        "Destroy diseased clumps.",
                        "Can cause up to 30% yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Sugarcane_Mosaic":
                imageView.setImageResource(R.drawable.sugarcane_mosaic);
                setInfo(
                        "Sugarcane Mosaic Virus",
                        "Viral disease affecting sugarcane leaves.",
                        "Caused by Sugarcane mosaic virus.",
                        "Mosaic yellow-green patterns on leaves.",
                        "Plant resistant varieties.",
                        "No direct chemical control (vector control needed).",
                        "Remove infected plants.",
                        "Losses up to 20–30%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Sugarcane_Red_Rot":
                imageView.setImageResource(R.drawable.sugarcane_red);
                setInfo(
                        "Sugarcane Red Rot",
                        "One of the most destructive fungal diseases.",
                        "Caused by Colletotrichum falcatum.",
                        "Reddish discoloration inside stalk, foul smell.",
                        "Use resistant varieties.",
                        "Apply Carbendazim or Propiconazole (recommended dose).",
                        "Crop rotation and field sanitation.",
                        "Yield losses up to 70%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Sugarcane_Rust":
                imageView.setImageResource(R.drawable.sugarcane_rust);
                setInfo(
                        "Sugarcane Rust",
                        "Fungal disease forming pustules on leaves.",
                        "Caused by Puccinia melanocephala.",
                        "Rusty brown pustules on lower leaf surfaces.",
                        "Plant resistant cultivars.",
                        "Use fungicides like Mancozeb or Triazoles.",
                        "Remove severely infected leaves.",
                        "Losses 10–40%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Sugarcane_Yellow":
                imageView.setImageResource(R.drawable.sugarcane_yellow);
                setInfo(
                        "Sugarcane Yellow Leaf Disease",
                        "Viral disease transmitted by aphids.",
                        "Caused by Sugarcane yellow leaf virus.",
                        "Midrib yellowing progressing to whole leaf.",
                        "Control aphid population.",
                        "No direct cure, manage with insecticides.",
                        "Use virus-free planting material.",
                        "Moderate yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Sugarcane_Healthy":
                imageView.setImageResource(R.drawable.sugarcane_healthy );
                setInfo(
                        "Healthy Sugarcane",
                        "No disease detected.",
                        "-",
                        "-",
                        "Continue good practices.",
                        "-",
                        "Maintain crop hygiene.",
                        "No yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;


            case "Rice_Tungro":
                imageView.setImageResource(R.drawable.rice_tungro); // Add image in drawable
                setInfo(
                        "Rice Tungro",
                        "Viral disease causing stunted growth and yellow-orange discoloration.",
                        "Caused by Rice tungro bacilliform virus and Rice tungro spherical virus transmitted by leafhoppers.",
                        "Yellowing of leaves, stunted plant growth.",
                        "Use resistant varieties and control leafhopper population.",
                        "No direct chemical treatment; manage vector population.",
                        "Remove infected plants.",
                        "May reduce yield by 50-70%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Sheath_Blight":
                imageView.setImageResource(R.drawable.rice_blight);
                setInfo(
                        "Rice Sheath Blight",
                        "Fungal disease affecting rice sheath and leaves.",
                        "Caused by Rhizoctonia solani.",
                        "Lesions on leaf sheaths, spreading to leaves, reduced tillering.",
                        "Ensure proper spacing and water management.",
                        "Use valid fungicides (e.g., Hexaconazole, Propiconazole).",
                        "Crop rotation with non-host crops.",
                        "Yield loss may reach 20-50%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Leaf_Scald":
                imageView.setImageResource(R.drawable.rice_scar);
                setInfo(
                        "Rice Leaf Scald",
                        "Fungal disease causing white to grey lesions on leaves.",
                        "Caused by Microdochium oryzae.",
                        "White-grey streaks on leaves with yellow borders.",
                        "Maintain proper water and nutrient management.",
                        "Apply appropriate fungicides.",
                        "Remove affected plant debris.",
                        "May reduce yield moderately.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Healthy":
                imageView.setImageResource(R.drawable.rice_healthy);
                setInfo(
                        "Healthy Rice Plant",
                        "No disease detected.",
                        "-",
                        "-",
                        "Continue good agronomic practices.",
                        "-",
                        "Maintain proper irrigation and fertilization.",
                        "No yield loss.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Brown_Spot":
                imageView.setImageResource(R.drawable.rice_brown);
                setInfo(
                        "Rice Brown Spot",
                        "Fungal disease causing brown lesions on leaves and grains.",
                        "Caused by Bipolaris oryzae.",
                        "Brown spots on leaves, can reduce grain quality.",
                        "Ensure balanced fertilization, proper spacing.",
                        "Apply recommended fungicides.",
                        "Remove infected residues.",
                        "Moderate yield reduction.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Blast":
                imageView.setImageResource(R.drawable.rice_blast);
                setInfo(
                        "Rice Blast",
                        "Serious fungal disease affecting leaves, nodes, and panicles.",
                        "Caused by Magnaporthe oryzae.",
                        "Diamond-shaped lesions on leaves, neck rot, grain sterility.",
                        "Use resistant varieties and avoid excessive nitrogen.",
                        "Fungicides like Tricyclazole may be used.",
                        "Crop rotation and remove infected debris.",
                        "Yield loss can be 50-70%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            case "Rice_Bacterial_Leaf_Blight":
                imageView.setImageResource(R.drawable.rice_leaf);
                setInfo(
                        "Rice Bacterial Leaf Blight",
                        "Bacterial disease causing yellowing and drying of leaf margins.",
                        "Caused by Xanthomonas oryzae pv. oryzae.",
                        "Yellowing of leaf edges, wilting, reduced tillers.",
                        "Use resistant varieties, avoid high nitrogen application.",
                        "Apply Copper-based bactericides.",
                        "Remove infected plant debris.",
                        "Yield loss may reach 20-50%.",
                        "Call Kisan Helpline 1800-180-1551"
                );
                break;

            // ---------- Default ----------
            default:
                setInfo("Unknown Disease", "No data available.", "-", "-", "-", "-", "-", "-", "-");
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


            translationHelper.translateTextView(lblDescription, "Description");
            translationHelper.translateTextView(lblCauses, "Causes");
            translationHelper.translateTextView(lblSymptoms, "Symptoms");
            translationHelper.translateTextView(lblPrecautions, "Precautions");
            translationHelper.translateTextView(lblPesticides, "Recommended Pesticides / Fungicides");

            translationHelper.translateTextView(card_text2, "Tagmycin 6gm");
            translationHelper.translateTextView(card_text3, "Syngenta Amistar Top Fungicide (100 ml)");
            translationHelper.translateTextView(card_text4, "Anoka Trichoderma + Pseudomonas Bio-control");
            translationHelper.translateTextView(card_text5, "Hexaconazole 4% Contact &amp; Systemic Fungicide");

            translationHelper.translateTextView(b1, "Buy");
            translationHelper.translateTextView(b2, "Buy");
            translationHelper.translateTextView(b3, "Buy");
            translationHelper.translateTextView(b4, "Buy");

            translationHelper.translateTextView(lblCultural, "Cultural Practices");
            translationHelper.translateTextView(lblCostLoss, "Cost and Expected Loss");
            translationHelper.translateTextView(lblHelpline, "Nearby Help / Helpline");
            translationHelper.translateTextView(call, "Call Help");
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
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        if (translationHelper != null) {
            translationHelper.close();
        }

    }



}