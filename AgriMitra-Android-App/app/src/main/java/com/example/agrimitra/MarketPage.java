package com.example.agrimitra;

import android.content.Intent;
import android.os.Bundle;
import android.widget.TextView;

import androidx.cardview.widget.CardView;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

/**
 * A simple {@link Fragment} subclass.
 * Use the {@link MarketPage#newInstance} factory method to
 * create an instance of this fragment.
 */
public class MarketPage extends Fragment {

    // TODO: Rename parameter arguments, choose names that match
    // the fragment initialization parameters, e.g. ARG_ITEM_NUMBER
    private static final String ARG_PARAM1 = "param1";
    private static final String ARG_PARAM2 = "param2";

    // TODO: Rename and change types of parameters
    private String mParam1;
    private String mParam2;
    
    private TranslationHelper translationHelper;

    public MarketPage() {
        // Required empty public constructor
    }

    /**
     * Use this factory method to create a new instance of
     * this fragment using the provided parameters.
     *
     * @param param1 Parameter 1.
     * @param param2 Parameter 2.
     * @return A new instance of fragment MarketPage.
     */
    // TODO: Rename and change types and number of parameters
    public static MarketPage newInstance(String param1, String param2) {
        MarketPage fragment = new MarketPage();
        Bundle args = new Bundle();
        args.putString(ARG_PARAM1, param1);
        args.putString(ARG_PARAM2, param2);
        fragment.setArguments(args);
        return fragment;
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (getArguments() != null) {
            mParam1 = getArguments().getString(ARG_PARAM1);
            mParam2 = getArguments().getString(ARG_PARAM2);
        }
        
        // Initialize translation helper
        translationHelper = new TranslationHelper(requireContext());
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        View view=inflater.inflate(R.layout.fragment_market_page, container, false);

        CardView live=view.findViewById(R.id.c1);
        CardView news=view.findViewById(R.id.c2);
        CardView tools=view.findViewById(R.id.c3);

        // Find TextViews for translation
        TextView cardText1 = view.findViewById(R.id.card_text1);
        TextView cardText2 = view.findViewById(R.id.card_text2);
        TextView cardText3 = view.findViewById(R.id.card_text3);

        // Initialize translation and translate card titles
        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(cardText1, "Live Market Prices");
            translationHelper.translateTextView(cardText2, "Agricultural News");
            translationHelper.translateTextView(cardText3, "Farm Tools & Equipment");
        });

        live.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getActivity(),MarketPriceActivity.class));
            }
        });

        news.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getActivity(),NewsPage.class));
            }
        });

        tools.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(getActivity(),MachinePage.class));
            }
        });


        return view;

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}