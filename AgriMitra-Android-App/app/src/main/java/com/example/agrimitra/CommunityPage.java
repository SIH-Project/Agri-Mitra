package com.example.agrimitra;

import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.cardview.widget.CardView;
import androidx.fragment.app.Fragment;

public class CommunityPage extends Fragment {

    private MLKitTranslator translator;
    private String selectedLang;

    public CommunityPage() {
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

        View view = inflater.inflate(R.layout.fragment_community_page, container, false);

        // Find CardViews and TextViews
        CardView community = view.findViewById(R.id.c1);
        CardView scheme = view.findViewById(R.id.c2);
        CardView bot = view.findViewById(R.id.c3);

        TextView cardText1 = view.findViewById(R.id.card_text1);
        TextView cardText2 = view.findViewById(R.id.card_text2);
        TextView cardText3 = view.findViewById(R.id.card_text3);

        // Translate card titles if needed
        if (translator != null) {
            translator.downloadModelIfNeeded(
                    () -> { // On success
                        translateCardText(cardText1, "Communities");
                        translateCardText(cardText2, "Government Schemes & Subsidies");
                        translateCardText(cardText3, "Multilingual Chatbot");
                    },
                    () -> {} // On failure, keep English
            );
        }

        // Set click listeners
        community.setOnClickListener(v -> startActivity(new Intent(getActivity(), CommunityMenu.class)));
        scheme.setOnClickListener(v -> startActivity(new Intent(getActivity(), Schemes.class)));
        bot.setOnClickListener(v -> startActivity(new Intent(getActivity(), ChatBot.class)));

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
