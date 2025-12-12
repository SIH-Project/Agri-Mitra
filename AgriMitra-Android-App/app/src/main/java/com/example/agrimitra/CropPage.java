package com.example.agrimitra;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.cardview.widget.CardView;
import androidx.fragment.app.Fragment;

public class CropPage extends Fragment {

    private MLKitTranslator translator;
    private String selectedLang;



    public CropPage() {
        // Required empty public constructor
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Get saved language
        SharedPreferences prefs = requireActivity().getSharedPreferences("AppPrefs", getActivity().MODE_PRIVATE);
        selectedLang = prefs.getString("SelectedLanguage", "en");

        // Initialize translator if not English
        if (!selectedLang.equals("en")) {
            translator = new MLKitTranslator();
            translator.initTranslator("en", selectedLang);
        }
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {

        View view = inflater.inflate(R.layout.fragment_crop_page, container, false);

        // Find all CardViews and their TextViews
        CardView pedia = view.findViewById(R.id.c6);
        CardView game = view.findViewById(R.id.c7);
        CardView book = view.findViewById(R.id.c8);
        CardView irrigation = view.findViewById(R.id.c4);
        CardView recommendation = view.findViewById(R.id.c5);
        CardView disease = view.findViewById(R.id.c3);
        CardView yield = view.findViewById(R.id.c1);
        CardView dashboard = view.findViewById(R.id.dashboard);
        CardView satellite = view.findViewById(R.id.c2);



        TextView cardText1 = view.findViewById(R.id.card_text1);
        TextView cardText2 = view.findViewById(R.id.card_text2);
        TextView cardText3 = view.findViewById(R.id.card_text3);
        TextView cardText4 = view.findViewById(R.id.card_text4);
        TextView cardText5 = view.findViewById(R.id.card_text5);
        TextView cardText6 = view.findViewById(R.id.card_text6);
        TextView cardText7 = view.findViewById(R.id.card_text7);
        TextView cardText8 = view.findViewById(R.id.card_text8);
        TextView mainTitle = view.findViewById(R.id.mainTitle);
        TextView dashboardtitle = view.findViewById(R.id.card_dashboard2);

        ImageView imageView= view.findViewById(R.id.agrimam);

        imageView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getActivity(), AgriMam.class));
            }
        });

        TextView t1 = view.findViewById(R.id.t1);


        CardView fertiCheck = view.findViewById(R.id.c9);
        TextView cardText9 = view.findViewById(R.id.card_text10);

        fertiCheck.setOnClickListener(v -> startActivity(new Intent(getActivity(), FertiCheckActivity.class)));

        // Translate card titles if needed
        if (translator != null) {
            translator.downloadModelIfNeeded(
                    () -> { // On success
                        translateCardText(cardText1, "Yield Prediction");
                        translateCardText(cardText2, "Satellite Analysis");
                        translateCardText(cardText3, "Crop Doctor\n(Disease Prediction)");
                        translateCardText(cardText4, "Irrigation Plan\n(Weekly)");
                        translateCardText(cardText5, "Crop Recommendation");
                        translateCardText(cardText6, "Fertilizer and Pesticide Pedia");
                        translateCardText(cardText7, "Simulation\nGame");
                        translateCardText(cardText8, "Farm Book");
                        translateCardText(cardText9, "Fertilizer Check");
                        translateCardText(dashboardtitle, "Farmer Dashboard");
                        translateCardText(mainTitle, "Crop Management");
                        translateCardText(t1, "Helpers");


                        },
                    () -> {} // Failure fallback, keep original English
            );
        }

        // Set click listeners
        yield.setOnClickListener(v -> startActivity(new Intent(getActivity(), YieldPrediction.class)));
        dashboard.setOnClickListener(v -> startActivity(new Intent(getActivity(), Dashboard.class)));
        satellite.setOnClickListener(v -> startActivity(new Intent(getActivity(), SatelliteAnalysis.class)));
        disease.setOnClickListener(v -> startActivity(new Intent(getActivity(), CropDisease.class)));
        recommendation.setOnClickListener(v -> startActivity(new Intent(getActivity(), CropRecommendation.class)));
        irrigation.setOnClickListener(v -> startActivity(new Intent(getActivity(), IrrigationActivity.class)));
        pedia.setOnClickListener(v -> startActivity(new Intent(getActivity(), Pedia.class)));
        game.setOnClickListener(v -> startActivity(new Intent(getActivity(), SimulatorActivity.class)));
        book.setOnClickListener(v -> startActivity(new Intent(getActivity(), FarmBookPage.class)));

        return view;
    }

    private void translateCardText(TextView textView, String originalText) {
        translator.translateText(originalText, translatedText -> textView.setText(translatedText));
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (translator != null) translator.close();
    }
}
