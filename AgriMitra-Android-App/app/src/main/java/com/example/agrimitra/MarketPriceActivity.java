package com.example.agrimitra;

import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.RelativeLayout;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.android.volley.VolleyError;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;



public class MarketPriceActivity extends AppCompatActivity {

    private TranslationHelper translationHelper;

    TextView t1, t2, button_text;

    Spinner spinnerCrop, spinnerState, spinnerDistrict;

    RelativeLayout loading;
    CardView btnFetch;
    TextView tvCrop, tvState, tvDistrict, tvPrice;
    String selectedCrop = "", selectedState = "", selectedDistrict = "";

    // Dummy crop list
    String[] crops = {"Select Crop","Wheat", "Rice", "Potato", "Onion", "Tomato"};

    // States
    String[] states = {"Select State","Odisha","Tamil Nadu" ,"Maharashtra", "Punjab"};
    int state=21;
    int crop=1;

    ImageView image;

    CardView card;
    Map<String, String[]> districtMap = new HashMap<>();

    TextView result;


    private static final String API_KEY = "563cfd05708d613bb9d8863fae713110ffd7ad39530fd1413e1fbc45395c3726";
    private static final String URL = "https://api.ceda.ashoka.edu.in/v1/agmarknet/prices";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_market_price);

        // Initialize translation helper
        translationHelper = new TranslationHelper(this);

        // Initialize views
        spinnerCrop = findViewById(R.id.spinnerCrop);
        spinnerState = findViewById(R.id.spinnerState);
        spinnerDistrict = findViewById(R.id.spinnerDistrict);

        t1 = findViewById(R.id.t1);
        t2 = findViewById(R.id.t2);
        button_text = findViewById(R.id.button_text);

        loading=findViewById(R.id.loading);
        loading.setVisibility(View.GONE);

        image=findViewById(R.id.card_image1);


        card=findViewById(R.id.c1);
        card.setVisibility(View.GONE);


        result=findViewById(R.id.card_text1);



        btnFetch = findViewById(R.id.btnFetch);





        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(t1, t1.getText().toString());
            translationHelper.translateTextView(t2, t2.getText().toString());
            translationHelper.translateTextView(button_text, button_text.getText().toString());

        });



        // Odisha districts
        districtMap.put("Odisha", new String[]{"Select District",
                "Bhubaneswar", "Cuttack", "Puri", "Sambalpur", "Balasore",
                "Berhampur", "Rourkela", "Jharsuguda", "Kendrapara", "Angul"
        });

// Maharashtra districts
        districtMap.put("Maharashtra", new String[]{"Select District",
                "Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad",
                "Thane", "Kolhapur", "Solapur", "Amravati", "Satara"
        });

// Punjab districts
        districtMap.put("Punjab", new String[]{"Select District",
                "Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda",
                "Hoshiarpur", "Gurdaspur", "Mohali", "Firozpur", "Sangrur"
        });

// Tamil Nadu districts
        districtMap.put("Tamil Nadu", new String[]{"Select District",
                "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
                "Tirunelveli", "Vellore", "Erode", "Thoothukudi", "Kanchipuram"
        });



        // Initialize translation and set up spinners
        translationHelper.initializeTranslation(() -> {
            ArrayAdapter<String> cropAdapter = new ArrayAdapter<>(this, R.layout.spinner_item, crops);
            cropAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerCrop.setAdapter(cropAdapter);
        });

        spinnerCrop.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                selectedCrop = crops[i];

                if (selectedCrop.equals("Wheat"))
                {
                    crop=1;
                    image.setImageResource(R.drawable.wheat);
                }
                else if (selectedCrop.equals("Paddy"))
                {
                    crop=2;
                    image.setImageResource(R.drawable.paddy);
                }
                else if (selectedCrop.equals("Rice"))
                {
                    crop=3;
                    image.setImageResource(R.drawable.rice);

                }
                else if (selectedCrop.equals("Maze"))
                {
                    crop=4;
                    image.setImageResource(R.drawable.maze);


                }
                else if (selectedCrop.equals("Groundnut"))
                {
                    crop=10;
                    image.setImageResource(R.drawable.groundnut);
                }
                else if (selectedCrop.equals("Soyabeen"))
                {
                    crop=13;
                    image.setImageResource(R.drawable.soyabean);
                }
                else if (selectedCrop.equals("Mango"))
                {
                    crop=20;
                    image.setImageResource(R.drawable.mango);
                }
                else if (selectedCrop.equals("Onion"))
                {
                    crop=23;
                    image.setImageResource(R.drawable.onion);
                }
                else if (selectedCrop.equals("Potato"))
                {
                    crop=24;
                    image.setImageResource(R.drawable.potato);
                }
                else if (selectedCrop.equals("Tomato"))
                {
                    crop=78;
                    image.setImageResource(R.drawable.potato);
                }












            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });


            ArrayAdapter<String> stateAdapter = new ArrayAdapter<>(this, R.layout.spinner_item, states);
            stateAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
            spinnerState.setAdapter(stateAdapter);

        spinnerState.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                selectedState = states[i];

                if (selectedState.equals("Odisha"))
                {
                    state=21;
                }
                else if (selectedState.equals("Tamil Nadu"))
                {
                    state=33;
                }
                else if (selectedState.equals("Rajasthan"))
                {
                    state=8;
                }
                else if (selectedState.equals("Delhi"))
                {
                    state=7;
                }
                else if (selectedState.equals("West Bengal"))
                {
                    state=19;
                }




                // Load only that state's districts
                String[] districts = districtMap.get(selectedState);
                if (districts != null) {
                    ArrayAdapter<String> districtAdapter = new ArrayAdapter<>(MarketPriceActivity.this,
                            R.layout.spinner_item, districts);
                    districtAdapter.setDropDownViewResource(R.layout.spinner_dropdown_item);
                    spinnerDistrict.setAdapter(districtAdapter);
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });

        spinnerDistrict.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
                String[] districts = districtMap.get(selectedState);
                if (districts != null) {
                    selectedDistrict = districts[i];
                }
            }
            @Override
            public void onNothingSelected(AdapterView<?> adapterView) {}
        });




        btnFetch.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                loading.setVisibility(View.VISIBLE);
                fetchMarketPrice();
            }
        });

        // Edge to edge padding
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }

    private void fetchMarketPrice() {

        String fromDate="2023-01-01";
        String toDate="2025-06-01";

        try {
            JSONObject jsonBody = new JSONObject();
            jsonBody.put("commodity_id", crop);
            jsonBody.put("state_id", state); // single integer
            jsonBody.put("district_id", new JSONArray());       // empty array
            jsonBody.put("market_id", new JSONArray());         // empty array
            jsonBody.put("from_date", fromDate);
            jsonBody.put("to_date", toDate);

            RequestQueue queue = Volley.newRequestQueue(this);

            JsonObjectRequest request = new JsonObjectRequest(Request.Method.POST, URL, jsonBody,
                    response -> parseResponse(response),
                    this::handleVolleyError) {
                @Override
                public Map<String, String> getHeaders() throws AuthFailureError {
                    Map<String, String> headers = new HashMap<>();
                    headers.put("Authorization", "Bearer " + API_KEY);
                    headers.put("Content-Type", "application/json");
                    return headers;
                }
            };

            queue.add(request);

        } catch (Exception e) {
            e.printStackTrace();
            translationHelper.showTranslatedToast("Error creating request");
        }
    }



    private void parseResponse(JSONObject response) {
        try {
            if (response != null && response.has("output")) {
                JSONObject output = response.getJSONObject("output");

                if (output.has("data")) {
                    JSONArray dataArray = output.getJSONArray("data");

                    if (dataArray.length() > 0) {
                        JSONObject data = dataArray.getJSONObject(0);

                        String currentDate = new SimpleDateFormat("dd-MM-yyyy", Locale.getDefault()).format(new Date());




                        String resultText = "Date: " + currentDate+"\nCrop: "+selectedCrop+"\nState: "+selectedState+"\nDistrict: "+selectedDistrict+"\nMin Price: "+ data.optInt("min_price", 0)
                        +"\nMax Price: "+ data.optInt("max_price", 0)+"\nModal Price: "+ data.optInt("modal_price", 0);
                        translationHelper.translateText(resultText, translatedText -> {
                            result.setText(translatedText);
                        });

                        loading.setVisibility(View.GONE);
                        card.setVisibility(View.VISIBLE);


                    } else {
                        translationHelper.showTranslatedToast("No data available");
                    }
                } else {
                    translationHelper.showTranslatedToast("API returned invalid structure");
                }
            } else {
                translationHelper.showTranslatedToast("API returned invalid structure");
            }
        } catch (Exception e) {
            e.printStackTrace();
            translationHelper.showTranslatedToast("Parsing error");
        }
    }


    private void handleVolleyError(VolleyError error) {
        String body = "";
        String statusCode = "";

        if (error.networkResponse != null) {
            statusCode = String.valueOf(error.networkResponse.statusCode);
            try {
                body = new String(error.networkResponse.data, "UTF-8");
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        Log.e("MarketPriceAPI", "Status Code: " + statusCode);
        Log.e("MarketPriceAPI", "Body: " + body);
        translationHelper.showTranslatedToast("API Error: " + statusCode + " " + body, Toast.LENGTH_LONG);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (translationHelper != null) {
            translationHelper.close();
        }
    }
}
