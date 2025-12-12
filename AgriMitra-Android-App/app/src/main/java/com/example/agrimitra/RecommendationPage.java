package com.example.agrimitra;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class RecommendationPage extends AppCompatActivity {

    private TranslationHelper translationHelper;
    ImageView poster,f1_image,f2_image;

    TextView title,analysis,nContent,nRequire,totalN,urea,can,as,dap,f1_text,f2_text,nContentText,totalTitle,secondCardTitle,thirdCardTitle,forthCardTitle,fifthCardTitle;

    CardView button1,button2;

    double delta=1.0;

    TextView  t1,t2,t7;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_recommendation_page);

        translationHelper = new TranslationHelper(this);

        t2 = findViewById(R.id.t2);
        t7 = findViewById(R.id.t7);









        poster=findViewById(R.id.img1);
        title=findViewById(R.id.t1);
        analysis=findViewById(R.id.t3);
        nContent=findViewById(R.id.t4);
        nContentText=findViewById(R.id.t5);
        nRequire=findViewById(R.id.t6);
        totalN=findViewById(R.id.t16);
        urea=findViewById(R.id.ut2);
        can=findViewById(R.id.ut4);
        as=findViewById(R.id.ut6);
        dap=findViewById(R.id.ut8);

        f1_image=findViewById(R.id.card_image2);
        f2_image=findViewById(R.id.card_image3);
        f1_text=findViewById(R.id.card_text2);
        f2_text=findViewById(R.id.card_text3);

        totalTitle=findViewById(R.id.t15);
        secondCardTitle=findViewById(R.id.ut1);
        thirdCardTitle=findViewById(R.id.ut3);
        forthCardTitle=findViewById(R.id.ut5);
        fifthCardTitle=findViewById(R.id.ut7);

        button1=findViewById(R.id.button1);
        button2=findViewById(R.id.button2);

        String topic=getIntent().getStringExtra("topic").toString().toLowerCase();
        String budget=getIntent().getStringExtra("budget").toString().toLowerCase();
//        int num=Integer.parseInt(budget);



        delta=getIntent().getDoubleExtra("delta",2.0);

        int num=24000;

        translationHelper.initializeTranslation(() -> {
            if (topic.equals("nitrogen"))
            {

                double totalNitrogen=delta*0.809;

                double total2=totalNitrogen/0.46;
                double total3=totalNitrogen/0.26;
                double total4=totalNitrogen/0.21;
                double total5=totalNitrogen/0.18;



                TextView ans1=findViewById(R.id.t16);
                TextView ans2=findViewById(R.id.ut2);
                TextView ans3=findViewById(R.id.ut4);
                TextView ans4=findViewById(R.id.ut6);
                TextView ans5=findViewById(R.id.ut8);

                ans1.setText(String.format("%.2f Kg/acre", totalNitrogen));
                ans2.setText(String.format("%.2f Kg/acre", total2));
                ans3.setText(String.format("%.2f Kg/acre", total3));
                ans4.setText(String.format("%.2f Kg/acre", total4));
                ans5.setText(String.format("%.2f Kg/acre", total5));











                translationHelper.translateTextView(t2, "Analysis");
                translationHelper.translateTextView(t7, "Fertilizer Recommendation\\n(Based on Budget)");

                poster.setImageResource(R.drawable.urea);
                translationHelper.translateTextView(title, "Nitrogen Fertilizer");
                translationHelper.translateTextView(analysis, "Rule: 1 ppm = 2Kg/ha ≈ 0.809Kg/acre\nΔ = 3.7 ppm → 3.0 kg N/acre");
                translationHelper.translateTextView(nContent, "Nitrogen Contents");
                translationHelper.translateTextView(nContentText, "Urea ≈ 46% N\nCAN ≈ 26% N\nAmmonium sulfate ≈ 21% N\nDAP ≈ 18% N");
                translationHelper.translateTextView(nRequire, "Nitrogen Requirements");
//                translationHelper.translateTextView(totalN, "3.0 Kg/acre");
//                translationHelper.translateTextView(urea, "Urea: 6.5 Kg/acre");
//                translationHelper.translateTextView(can, "CAN: 11.5 Kg/acre");
//                translationHelper.translateTextView(as, "Ammonium sulfate: 14.3 Kg/acre");
//                translationHelper.translateTextView(dap, "DAP: 16.6 Kg/acre");

                translationHelper.translateTextView(totalTitle, "Total Nitrogen");
                translationHelper.translateTextView(secondCardTitle, "Urea");
                translationHelper.translateTextView(thirdCardTitle, "CAN");
                translationHelper.translateTextView(forthCardTitle, "Ammonium Sulfate");
                translationHelper.translateTextView(fifthCardTitle, "DAP");


            if (num<25000)
            {
                f1_image.setImageResource(R.drawable.low_cost_f1);
                f2_image.setImageResource(R.drawable.low_cost_f2);
                translationHelper.translateTextView(f2_text, "Urea Fertilizer(250gm)\nPrice:Rs.79\nDiscount:23%");
                translationHelper.translateTextView(f1_text, "Plant Food Fertilizer Sticks\nPrice: Rs. 249\nDiscount: 56%");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Fertilizer-fertilizer-Multi-Purpose-Fertilizers-40/dp/B0FN89PV5V/ref=sr_1_1_sspa?dib=eyJ2IjoiMSJ9.CKzLPKwWkZaoRuyaXcPxXFPI9-uNj-eio8LPXna1E-c_chCs_BxVcAxqSr63EM8BGeA1Oj80CIrdz_2mH319EWVwuK9E1oVKpNhgWJT9cAwyA5q_E0skAz3JCF28kEUa6SIQwQVvnsUmfhYLhfxs1pxRTkOl-imSxMHXPpgKKSZhdXH8GXuLS__2GYk7h3489XkdvxmyWaxTOj_pciqJDTeNLiTWwd_Z7Ig_DKTFcTFCSIdKMQKhundFxDtvLuDDAhB2NVivS32QM6BT00A32ogGNGtIhgpuT8qmXvulec4.Wr8xckSF-6h0TyDwy6RMRnIZaPHr4WcoFL2hKrWdg8s&dib_tag=se&keywords=urea+for+plants&qid=1758453175&sr=8-1-spons&utm_source=chatgpt.com&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/Fertilizer-Soluble-Vegetables-Flowers-Production/dp/B0DFGYY2N1/ref=sr_1_6?dib=eyJ2IjoiMSJ9.CKzLPKwWkZaoRuyaXcPxXFPI9-uNj-eio8LPXna1E-c_chCs_BxVcAxqSr63EM8BGeA1Oj80CIrdz_2mH319EWVwuK9E1oVKpNhgWJT9cAwyA5q_E0skAz3JCF28kEUa6SIQwQVvnsUmfhYLhfxs1pxRTkOl-imSxMHXPpgKKSZhdXH8GXuLS__2GYk7h3489XkdvxmyWaxTOj_pciqJDTeNLiTWwd_Z7Ig_DKTFcTFCSIdKMQKhundFxDtvLuDDAhB2NVivS32QM6BT00A32ogGNGtIhgpuT8qmXvulec4.Wr8xckSF-6h0TyDwy6RMRnIZaPHr4WcoFL2hKrWdg8s&dib_tag=se&keywords=urea+for+plants&qid=1758453175&sr=8-6&utm_source=chatgpt.com"));


            }
            else if (num<60000)
            {
                f1_image.setImageResource(R.drawable.mid_cost_f1);
                f2_image.setImageResource(R.drawable.mid_cost_f2);
                translationHelper.translateTextView(f2_text, "Utkarsh Boronated CAN\nPrice:Rs.701\nDiscount:65%");
                translationHelper.translateTextView(f1_text, "Organic Plant Bio Organic DAP Fertilizer\nPrice: Rs. 1161\nDiscount: 42%");


                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Boronated-Fertilizer-Flowering-Fertilization-Pack/dp/B0DFVKLBVL?utm_source=chatgpt.com&th=1"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/kribhco-Urea-Fertilizer-Nitrogen-Vegetables/dp/B0F893FCWS?utm_source=chatgpt.com"));





            }
            else
            {
                f1_image.setImageResource(R.drawable.high_cost_f1);
                f2_image.setImageResource(R.drawable.high_cost_f2);
                translationHelper.translateTextView(f2_text, "Nitrogen Fixing Microbes, Liquid Bio Fertilizer\nPrice:Rs.1364\nDiscount:65%");
                translationHelper.translateTextView(f1_text, "NPK Fertilizers for Plants Growth & Healthy Home Garden \nPrice: Rs. 1016\nDiscount: 54%");

                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Utkarsh-Acetobacter-Fertilizer-Structure-Fertility/dp/B0D3LKL499/ref=sr_1_23_sspa?dib=eyJ2IjoiMSJ9.HTbdxo_pvN81eFb5PeVibE9lA_YqrsFsSe7-RwT6i_O2WC0RwG83UO1En2FIhiXfljM1JtFmU3TAE_CHvr-55vNdO54WY9qv0eGAc8IyErndEdZYMnTGIAxLpNI1zq64Ck__fq0L1uQ0bbHbLLsZjc07vJpsGoiBKSpSHnoX603d9chpXOPtsPdwFz_tWFXIDZpPkk8JMaL8kzpnCs9OxUVklAamtkAW-N2GK75nAm6OxKsi7Y9r1ULhMYEgA_9AP5LJKy2SsP3gZzD06ONL4KdNFFNi2PQka8u8FUKpvSY.0eVT91nagud6hbOXaa5vrRkwsEO3WtcwY29mMVOjFLc&dib_tag=se&keywords=nitrogen+fertilizer+for+crop&qid=1758453323&sr=8-23-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9tdGY&psc=1"));

                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/Foodicine-10-Kg-Bio-Weed/dp/B0CFLVWR24/ref=sr_1_58_sspa?dib=eyJ2IjoiMSJ9.HTbdxo_pvN81eFb5PeVibE9lA_YqrsFsSe7-RwT6i_O2WC0RwG83UO1En2FIhiXfljM1JtFmU3TAE_CHvr-55vNdO54WY9qv0eGAc8IyErndEdZYMnTGIAxLpNI1zq64Ck__fq0L1uQ0bbHbLLsZjc07vJpsGoiBKSpSHnoX603d9chpXOPtsPdwFz_tWFXIDZpPkk8JMaL8kzpnCs9OxUVklAamtkAW-N2GK75nAm6OxKsi7Y9r1ULhMYEgA_9AP5LJKy2SsP3gZzD06ONL4KdNFFNi2PQka8u8FUKpvSY.0eVT91nagud6hbOXaa5vrRkwsEO3WtcwY29mMVOjFLc&dib_tag=se&keywords=nitrogen%2Bfertilizer%2Bfor%2Bcrop&qid=1758453323&sr=8-58-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9idGY&th=1"));





            }




        }
        else if (topic.equals("phosphorus"))
        {



            double totalNitrogen=delta*0.809;

            double total2=totalNitrogen/0.46;
            double total3=totalNitrogen/0.16;
            double total4=totalNitrogen/0.46;
            double total5=totalNitrogen/0.30;



            TextView ans1=findViewById(R.id.t16);
            TextView ans2=findViewById(R.id.ut2);
            TextView ans3=findViewById(R.id.ut4);
            TextView ans4=findViewById(R.id.ut6);
            TextView ans5=findViewById(R.id.ut8);

            ans1.setText(String.format("%.2f Kg/acre", totalNitrogen));
            ans2.setText(String.format("%.2f Kg/acre", total2));
            ans3.setText(String.format("%.2f Kg/acre", total3));
            ans4.setText(String.format("%.2f Kg/acre", total4));
            ans5.setText(String.format("%.2f Kg/acre", total5));





            poster.setImageResource(R.drawable.dap);
            translationHelper.translateTextView(title, "Phosphorus Fertilizer");
            translationHelper.translateTextView(analysis, "Rule: 1 ppm = 2Kg/ha ≈ 0.809Kg/acre\nΔ Example: 4 ppm → 3.2 kg P/acre");
            translationHelper.translateTextView(nContent, "Phosphorus Contents");
            translationHelper.translateTextView(nContentText, "DAP ≈ 46% P2O5 (20% P)\nSSP ≈ 16% P2O5 (7% P)\nTSP ≈ 46% P2O5 (20% P)\nRock Phosphate ≈ 30% P2O5");
            translationHelper.translateTextView(nRequire, "Phosphorus Requirements");
//            translationHelper.translateTextView(totalN, "≈3.2 Kg/acre");
//            translationHelper.translateTextView(urea, "DAP: ~16 Kg/acre");
//            translationHelper.translateTextView(can, "SSP: ~45 Kg/acre");
//            translationHelper.translateTextView(as, "TSP: ~16 Kg/acre");
//            translationHelper.translateTextView(dap, "Rock Phosphate: ~25 Kg/acre");

            translationHelper.translateTextView(totalTitle, "Total Phosphorus");
            translationHelper.translateTextView(secondCardTitle, "DAP");
            translationHelper.translateTextView(thirdCardTitle, "SSP");
            translationHelper.translateTextView(forthCardTitle, "TSP");
            translationHelper.translateTextView(fifthCardTitle, "Rock Phosphate");

            // Budget based phosphorus fertilizers
            if (num < 25000) {
                f1_image.setImageResource(R.drawable.low_cost_f1);
                f2_image.setImageResource(R.drawable.low_cost_f2);
                translationHelper.translateTextView(f1_text, "DAP Fertilizer (1kg)\nAffordable & effective\nPrice: Rs. 120");
                translationHelper.translateTextView(f2_text, "SSP Fertilizer (2kg)\nEconomical phosphorus source\nPrice: Rs. 150");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/DAP-Fertilizer-1kg/dp/B0BXYZ1234"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/SSP-Fertilizer-2kg/dp/B0CXYZ5678"));
            } else if (num < 60000) {
                f1_image.setImageResource(R.drawable.mid_cost_f1);
                f2_image.setImageResource(R.drawable.mid_cost_f2);
                translationHelper.translateTextView(f1_text, "TSP Fertilizer (5kg)\nGood for medium farms\nPrice: Rs. 550");
                translationHelper.translateTextView(f2_text, "DAP Fertilizer (10kg)\nHigh quality phosphorus\nPrice: Rs. 1160");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/TSP-Fertilizer-5kg/dp/B0DXYZ3456"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/DAP-Fertilizer-10kg/dp/B0EXYZ7890"));
            } else {
                f1_image.setImageResource(R.drawable.high_cost_f1);
                f2_image.setImageResource(R.drawable.high_cost_f2);
                translationHelper.translateTextView(f1_text, "Organic Rock Phosphate (20kg)\nPremium phosphorus source\nPrice: Rs. 2500");
                translationHelper.translateTextView(f2_text, "Imported TSP Fertilizer (25kg)\nHigh yield crops\nPrice: Rs. 3200");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Organic-Rock-Phosphate-20kg/dp/B0FXYZ1357"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/Imported-TSP-Fertilizer-25kg/dp/B0GXYZ2468"));
            }
        }
        else if (topic.equals("potassium"))
        {


            double totalNitrogen=delta*0.809;

            double total2=totalNitrogen/0.60;
            double total3=totalNitrogen/0.50;
            double total4=totalNitrogen/0.44;
            double total5=totalNitrogen/0.065;



            TextView ans1=findViewById(R.id.t16);
            TextView ans2=findViewById(R.id.ut2);
            TextView ans3=findViewById(R.id.ut4);
            TextView ans4=findViewById(R.id.ut6);
            TextView ans5=findViewById(R.id.ut8);

            ans1.setText(String.format("%.2f Kg/acre", totalNitrogen));
            ans2.setText(String.format("%.2f Kg/acre", total2));
            ans3.setText(String.format("%.2f Kg/acre", total3));
            ans4.setText(String.format("%.2f Kg/acre", total4));
            ans5.setText(String.format("%.2f Kg/acre", total5));



            poster.setImageResource(R.drawable.mop);
            translationHelper.translateTextView(title, "Potassium Fertilizer");
            translationHelper.translateTextView(analysis, "Rule: 1 ppm = 2Kg/ha ≈ 0.809Kg/acre\nΔ Example: 5 ppm → 3.2 kg K/acre");
            translationHelper.translateTextView(nContent, "Potassium Contents");
            translationHelper.translateTextView(nContentText, "MOP ≈ 60% K2O (50% K)\nSOP ≈ 50% K2O (42% K)\nKNO3 ≈ 44% K\nWood Ash ≈ 5–7% K2O");
            translationHelper.translateTextView(nRequire, "Potassium Requirements");
//            translationHelper.translateTextView(totalN, "≈3.96 Kg/acre");
//            translationHelper.translateTextView(urea, "MOP: ~6.5 Kg/acre");
//            translationHelper.translateTextView(can, "SOP: ~7.92 Kg/acre");
//            translationHelper.translateTextView(as, "KNO3: ~9 Kg/acre");
//            translationHelper.translateTextView(dap, "Wood Ash: ~56 Kg/acre");

            translationHelper.translateTextView(totalTitle, "Total Potassium");
            translationHelper.translateTextView(secondCardTitle, "MOP");
            translationHelper.translateTextView(thirdCardTitle, "SOP");
            translationHelper.translateTextView(forthCardTitle, "KNO3");
            translationHelper.translateTextView(fifthCardTitle, "Wood Ash");

            // Budget based potassium fertilizers
            if (num < 25000) {
                f1_image.setImageResource(R.drawable.low_cost_f1);
                f2_image.setImageResource(R.drawable.low_cost_f2);
                translationHelper.translateTextView(f1_text, "MOP (1kg)\nAffordable & effective\nPrice: Rs. 100");
                translationHelper.translateTextView(f2_text, "SOP (1kg)\nEconomical potassium source\nPrice: Rs. 120");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/MOP-Potash-1kg/dp/B0HXYZ1234"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/SOP-Potash-1kg/dp/B0JXYZ5678"));
            } else if (num < 60000) {
                f1_image.setImageResource(R.drawable.mid_cost_f1);
                f2_image.setImageResource(R.drawable.mid_cost_f2);
                translationHelper.translateTextView(f1_text, "KNO3 (5kg)\nGood for medium farms\nPrice: Rs. 600");
                translationHelper.translateTextView(f2_text, "MOP (10kg)\nHigh quality potassium\nPrice: Rs. 850");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Potassium-Nitrate-KNO3-5kg/dp/B0KXYZ3456"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/MOP-Fertilizer-10kg/dp/B0LXYZ7890"));
            } else {
                f1_image.setImageResource(R.drawable.high_cost_f1);
                f2_image.setImageResource(R.drawable.high_cost_f2);
                translationHelper.translateTextView(f1_text, "Organic SOP (20kg)\nPremium potassium source\nPrice: Rs. 2000");
                translationHelper.translateTextView(f2_text, "Potassium Sulfate MOP (25kg)\nHigh yield crops\nPrice: Rs. 2800");
                button1.setOnClickListener(v -> openUrl("https://www.amazon.in/Organic-SOP-Fertilizer-20kg/dp/B0MXYZ1357"));
                button2.setOnClickListener(v -> openUrl("https://www.amazon.in/Potassium-Sulfate-MOP-25kg/dp/B0NXYZ2468"));
            }
        }
            else if (topic.equals("moisture"))
            {
                poster.setImageResource(R.drawable.irrigation); // Add a relevant image in drawable
                translationHelper.translateTextView(title, "Moisture Management");
                translationHelper.translateTextView(analysis, "Rule: Optimal soil moisture is 60–70% of field capacity.");
                translationHelper.translateTextView(nContent, "Moisture Contents");
                translationHelper.translateTextView(nContentText, "Too low → plant wilting\nToo high → root rot, fungal attack");
                translationHelper.translateTextView(nRequire, "Moisture Requirement");
                translationHelper.translateTextView(totalN, "Maintain 60–70% moisture");

                translationHelper.translateTextView(urea, "Drip Irrigation System");
                translationHelper.translateTextView(can, "Mulching Sheets");
                translationHelper.translateTextView(as, "Soil Moisture Sensors");
                translationHelper.translateTextView(dap, "Organic Mulch");

                translationHelper.translateTextView(totalTitle, "Moisture Control Tools");
                translationHelper.translateTextView(secondCardTitle, "Drip Irrigation");
                translationHelper.translateTextView(thirdCardTitle, "Mulching Sheets");
                translationHelper.translateTextView(forthCardTitle, "Moisture Sensors");
                translationHelper.translateTextView(fifthCardTitle, "Organic Mulch");

                // Budget-based recommendations
                if (num < 25000) {
                    f1_image.setImageResource(R.drawable.low_cost_f1);
                    f2_image.setImageResource(R.drawable.low_cost_f2);
                    translationHelper.translateTextView(f1_text, "Plastic Mulching Sheet (1kg)\nPrice: Rs. 199");
                    translationHelper.translateTextView(f2_text, "Soil Moisture Tester\nPrice: Rs. 399");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=mulching+sheet"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=soil+moisture+meter"));
                } else if (num < 60000) {
                    f1_image.setImageResource(R.drawable.mid_cost_f1);
                    f2_image.setImageResource(R.drawable.mid_cost_f2);
                    translationHelper.translateTextView(f1_text, "Mini Drip Irrigation Kit\nPrice: Rs. 2200");
                    translationHelper.translateTextView(f2_text, "Soil Moisture Sensor Kit\nPrice: Rs. 1800");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=drip+irrigation+kit"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=soil+moisture+sensor"));
                } else {
                    f1_image.setImageResource(R.drawable.high_cost_f1);
                    f2_image.setImageResource(R.drawable.high_cost_f2);
                    translationHelper.translateTextView(f1_text, "Automatic Drip Irrigation System\nPrice: Rs. 6500");
                    translationHelper.translateTextView(f2_text, "Digital Soil Moisture & pH Tester\nPrice: Rs. 5200");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=automatic+drip+irrigation+system"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=digital+soil+moisture+ph+meter"));
                }
            }
            else if (topic.equals("ph"))
            {
                poster.setImageResource(R.drawable.irrigation); // Add a relevant image in drawable
                translationHelper.translateTextView(title, "Soil pH Management");
                translationHelper.translateTextView(analysis, "Rule: Optimal soil pH is 7.05–7.15 for most crops.");
                translationHelper.translateTextView(nContent, "pH Impact");
                translationHelper.translateTextView(nContentText, "Low pH (acidic) → nutrient lock\nHigh pH (alkaline) → micronutrient deficiency");
                translationHelper.translateTextView(nRequire, "pH Adjustment Needs");
                translationHelper.translateTextView(totalN, "Target pH: 6.5–7.0");

                translationHelper.translateTextView(urea, "Agricultural Lime (raises pH)");
                translationHelper.translateTextView(can, "Dolomite Powder (raises pH, adds Ca+Mg)");
                translationHelper.translateTextView(as, "Sulfur Powder (lowers pH)");
                translationHelper.translateTextView(dap, "Gypsum (improves alkaline soils)");

                translationHelper.translateTextView(totalTitle, "Soil Amendments");
                translationHelper.translateTextView(secondCardTitle, "Lime");
                translationHelper.translateTextView(thirdCardTitle, "Dolomite");
                translationHelper.translateTextView(forthCardTitle, "Sulfur");
                translationHelper.translateTextView(fifthCardTitle, "Gypsum");

                // Budget-based recommendations
                if (num < 25000) {
                    f1_image.setImageResource(R.drawable.low_cost_f1);
                    f2_image.setImageResource(R.drawable.low_cost_f2);
                    translationHelper.translateTextView(f1_text, "Garden Lime Powder (5kg)\nPrice: Rs. 300");
                    translationHelper.translateTextView(f2_text, "Sulfur Powder (2kg)\nPrice: Rs. 450");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=garden+lime+powder"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=sulfur+powder+fertilizer"));
                } else if (num < 60000) {
                    f1_image.setImageResource(R.drawable.mid_cost_f1);
                    f2_image.setImageResource(R.drawable.mid_cost_f2);
                    translationHelper.translateTextView(f1_text, "Dolomite Powder (25kg)\nPrice: Rs. 1600");
                    translationHelper.translateTextView(f2_text, "Agricultural Gypsum (50kg)\nPrice: Rs. 2200");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=dolomite+powder+fertilizer"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=gypsum+fertilizer"));
                } else {
                    f1_image.setImageResource(R.drawable.high_cost_f1);
                    f2_image.setImageResource(R.drawable.high_cost_f2);
                    translationHelper.translateTextView(f1_text, "Premium Lime Granules (50kg)\nPrice: Rs. 4500");
                    translationHelper.translateTextView(f2_text, "Soil Conditioner with Gypsum & Sulfur (25kg)\nPrice: Rs. 5200");
                    button1.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=lime+granules+fertilizer"));
                    button2.setOnClickListener(v -> openUrl("https://www.amazon.in/s?k=soil+conditioner+gypsum+sulfur"));
                }
            }




        });

        // keep same f1_text and f2_text logic


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



    private void openUrl(String url) {
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url));
        startActivity(intent);
    }

}
