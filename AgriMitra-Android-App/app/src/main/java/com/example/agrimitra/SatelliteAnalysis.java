package com.example.agrimitra;

import android.app.ProgressDialog;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class SatelliteAnalysis extends AppCompatActivity {

    private TranslationHelper translationHelper;

    RelativeLayout relativeLayout2;
    int count=0;
    RelativeLayout relativeLayout1;
    ImageView img1,img2,img3;

    TextView t3,t4,submit_text,current_text,text2,t2,info_text;

    TextView text3,mt_1,text4,mt_2,text5,mt_3;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_satellite_analysis);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        img1=findViewById(R.id.image1);
        img2=findViewById(R.id.image2);
        img3=findViewById(R.id.image3);


        t3 = findViewById(R.id.t3);
        t4 = findViewById(R.id.t4);
        submit_text = findViewById(R.id.submit_text);
        current_text = findViewById(R.id.current_text);
        text2 = findViewById(R.id.text2);
        t2 = findViewById(R.id.t2);
        info_text = findViewById(R.id.info_text);
        text3 = findViewById(R.id.text3);
        mt_1 = findViewById(R.id.mt_1);
        text4 = findViewById(R.id.text4);
        mt_2 = findViewById(R.id.mt_2);
        text5 = findViewById(R.id.text5);
        mt_3 = findViewById(R.id.mt_3);



        relativeLayout1=findViewById(R.id.rl1);
        relativeLayout2=findViewById(R.id.rl2);

        relativeLayout2.setVisibility(View.GONE);
        relativeLayout1.setVisibility(View.GONE);


        CardView submit=findViewById(R.id.cv1);

        translationHelper.initializeTranslation(() -> {

            translationHelper.translateTextView(t3, "Longitude");
            translationHelper.translateTextView(t4, "Latitude");
            translationHelper.translateTextView(submit_text, "Submit");
            translationHelper.translateTextView(current_text, "Current Values");
            translationHelper.translateTextView(text2, "Results");
            translationHelper.translateTextView(t2, "Farming Statistics");
            translationHelper.translateTextView(info_text, "Coverage Area: 2500 ha\n" +
                    "Data Quality: cloud_masked\n\n" +
                    "NDVI Mean: 0.3376318325\n\n" +
                    "NDVI Std: 0.16214700\n\n" +
                    "Region: Tamil Nadu\n\n" +
                    "Growth Stage: Stage 1(seeding)\n\n" +
                    "Management: Ensure adequate soil moisture,monitor for pest");


            translationHelper.translateTextView(text3, "NDVI");
            translationHelper.translateTextView(mt_1, "More Green = Dense, healthy, actively photosynthesizing crops (more chlorophyll).");
            translationHelper.translateTextView(text4, "NDRE");
            translationHelper.translateTextView(mt_2, "More Green = More Nitrogen Rich Soil.");
            translationHelper.translateTextView(text5, "Nutrients Map");
            translationHelper.translateTextView(mt_3, "More Red = More nutrients deficient area.");








        });



        submit.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                ProgressDialog progressDialog = new ProgressDialog(SatelliteAnalysis.this);
                translationHelper.translateText("Fetching data...", translatedText -> {
                    progressDialog.setMessage(translatedText);
                });
                progressDialog.setCancelable(false);
                progressDialog.show();

                new Handler().postDelayed(() -> {
                    progressDialog.dismiss();

                    // Static values (only numbers, no units)

                    relativeLayout2.setVisibility(View.GONE);
                    relativeLayout1.setVisibility(View.VISIBLE);


                    if (count%4==0)
                    {
                        img1.setImageResource(R.drawable.ndv1);
                        img2.setImageResource(R.drawable.ndre1);
                        img3.setImageResource(R.drawable.growth1);

                        info_text.setText("Coverage Area: 2500 ha\n" +
                                "Data Quality: cloud_masked\n\n" +
                                "NDVI Mean: 0.3376318325\n\n" +
                                "NDVI Std: 0.16214700\n\n" +
                                "Region: Tamil Nadu\n\n" +
                                "Growth Stage: Stage 1(seeding)\n\n" +
                                "Management: Ensure adequate soil moisture,monitor for pest");


                        translationHelper.initializeTranslation(() -> {

                            translationHelper.translateTextView(info_text, info_text.getText().toString());


                        });


                    }
                    else if (count%4==1)
                    {
                        img1.setImageResource(R.drawable.ndvi2);
                        img2.setImageResource(R.drawable.ndre2);
                        img3.setImageResource(R.drawable.growth2);

                        info_text.setText("Coverage Area: 2500 ha\n" +
                                "Data Quality: cloud_masked\n\n" +
                                "NDVI Mean: 0.37743782\n\n" +
                                "NDVI Std: 0.1972893\n\n" +
                                "Region: Tamil Nadu\n\n" +
                                "Growth Stage: Stage 1(seeding)\n\n" +
                                "Management: Ensure adequate soil nitrogen value,monitor irrigation");

                        translationHelper.initializeTranslation(() -> {

                            translationHelper.translateTextView(info_text, info_text.getText().toString());


                        });


                    }
                    else if (count%4==2)
                    {
                        img1.setImageResource(R.drawable.ndvi3);
                        img2.setImageResource(R.drawable.ndre3);
                        img3.setImageResource(R.drawable.growth3);

                        info_text.setText("Coverage Area: 2500 ha\n" +
                                "Data Quality: cloud_masked\n\n" +
                                "NDVI Mean: 0.57289903\n\n" +
                                "NDVI Std: 0.259392\n\n" +
                                "Region: Tamil Nadu\n\n" +
                                "Growth Stage: Stage 1(seeding)\n\n" +
                                "Management: Nitrogen value is good,monitor irrigation");

                        translationHelper.initializeTranslation(() -> {

                            translationHelper.translateTextView(info_text, info_text.getText().toString());


                        });


                    }
                    else if (count%4==3)
                    {
                        img1.setImageResource(R.drawable.ndvi4);
                        img2.setImageResource(R.drawable.ndri4);
                        img3.setImageResource(R.drawable.growth4);

                        info_text.setText("Coverage Area: 2500 ha\n" +
                                "Data Quality: cloud_masked\n\n" +
                                "NDVI Mean: 0.17289903\n\n" +
                                "NDVI Std: 0.059392\n\n" +
                                "Region: Tamil Nadu\n\n" +
                                "Growth Stage: Stage 1(seeding)\n\n" +
                                "Management: Nitrogen value is bad, apply good nitrogen fertilizers");

                        translationHelper.initializeTranslation(() -> {

                            translationHelper.translateTextView(info_text, info_text.getText().toString());


                        });



                    }
                    else
                    {
                        img1.setImageResource(R.drawable.ndv1);
                        img2.setImageResource(R.drawable.ndre1);
                        img3.setImageResource(R.drawable.growth1);

                        info_text.setText("Coverage Area: 2500 ha\n" +
                                "Data Quality: cloud_masked\n\n" +
                                "NDVI Mean: 0.38971237\n\n" +
                                "NDVI Std: 0.190238\n\n" +
                                "Region: Tamil Nadu\n\n" +
                                "Growth Stage: Stage 1(seeding)\n\n" +
                                "Management: Nitrogen value is good, take care of soil nutrients");

                        translationHelper.initializeTranslation(() -> {

                            translationHelper.translateTextView(info_text, info_text.getText().toString());


                        });


                    }

                    count++;

                }, 2000);







            }
        });



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
    }
}