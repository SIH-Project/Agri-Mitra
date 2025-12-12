    package com.example.agrimitra;

    import android.content.SharedPreferences;
    import android.os.Bundle;
    import android.view.View;

    import androidx.activity.EdgeToEdge;
    import androidx.appcompat.app.AppCompatActivity;
    import androidx.cardview.widget.CardView;
    import androidx.core.graphics.Insets;
    import androidx.core.view.ViewCompat;
    import androidx.core.view.WindowInsetsCompat;
    import android.content.Intent;
    import android.os.Bundle;
    import android.speech.RecognizerIntent;
    import android.speech.tts.TextToSpeech;
    import android.view.View;
    import android.widget.Toast;

    import androidx.activity.EdgeToEdge;
    import androidx.appcompat.app.AppCompatActivity;
    import androidx.cardview.widget.CardView;
    import androidx.core.graphics.Insets;
    import androidx.core.view.ViewCompat;
    import androidx.core.view.WindowInsetsCompat;

    import com.android.volley.Request;
    import com.android.volley.RequestQueue;
    import com.android.volley.Response;
    import com.android.volley.VolleyError;
    import com.android.volley.toolbox.JsonObjectRequest;
    import com.android.volley.toolbox.Volley;

    import org.json.JSONArray;
    import org.json.JSONException;
    import org.json.JSONObject;

    import java.util.ArrayList;
    import java.util.Locale;

    public class AgriMam extends AppCompatActivity {

        private static final int REQ_CODE_SPEECH_INPUT = 100;

        CardView mic, go;
        TextToSpeech tts;
        RequestQueue requestQueue;




        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            EdgeToEdge.enable(this);
            setContentView(R.layout.activity_agri_mam);


            mic=findViewById(R.id.card_mic);
            go=findViewById(R.id.card_go);

            go.setVisibility(View.GONE);





            requestQueue = Volley.newRequestQueue(this);

            SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
            String selectedLang = prefs.getString("selectedLanguage", "en"); // default English


            tts = new TextToSpeech(this, status -> {
                if (status != TextToSpeech.ERROR) {

                    Locale locale;

                    // Map your stored language code to a Locale
                    switch (selectedLang) {
                        case "ta": // Tamil
                            locale = new Locale("ta", "IN");
                            break;
                        default: // English
                            locale = Locale.ENGLISH;
                    }

                    int result = tts.setLanguage(locale);

                    if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Toast.makeText(this, "Selected language not supported for TTS", Toast.LENGTH_SHORT).show();
                    }

                    tts.setPitch(1.0f);
                    tts.setSpeechRate(1.0f);
                }
            });






            mic.setOnClickListener(v -> startSpeechToText());



            ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
                Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
                v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
                return insets;
            });
        }

        private void startSpeechToText() {
            // Get the selected language code from SharedPreferences
            SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
            String selectedLang = prefs.getString("SelectedLanguage", "en"); // default English

            // Map language codes to proper locale strings and prompts
            String localeCode;
            String promptText;

            if (selectedLang.equals("ta")) {
                localeCode = "ta-IN"; // Tamil
                promptText = "உங்கள் கேள்வியை பேசுங்கள்";
            } else {
                localeCode = "en-IN"; // English (default)
                promptText = "Speak your question";
            }


            // Create intent for speech recognition
            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, localeCode);
            intent.putExtra(RecognizerIntent.EXTRA_PROMPT, promptText);

            try {
                startActivityForResult(intent, REQ_CODE_SPEECH_INPUT);
            } catch (Exception e) {
                Toast.makeText(this, "Speech recognition not supported for this language", Toast.LENGTH_SHORT).show();
            }
        }




        @Override
        protected void onActivityResult(int requestCode, int resultCode, Intent data) {
            super.onActivityResult(requestCode, resultCode, data);

            if (requestCode == REQ_CODE_SPEECH_INPUT && resultCode == RESULT_OK && data != null) {
                ArrayList<String> result = data.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                if (result != null && !result.isEmpty()) {
                    String userQuery = result.get(0);
                    sendQueryToModel(userQuery);
                }
            }
        }

        private void sendQueryToModel(String query) {



            SharedPreferences p = getSharedPreferences("AppPrefs", MODE_PRIVATE);
            String sl = p.getString("SelectedLanguage", "en"); // default English

            // 2️⃣ Choose API endpoint based on language
            String url;
            if (sl.equals("ta")) {
                // Tamil model endpoint (replace this with your Tamil model URL)
                url = "https://tamilagentfinal.onrender.com/api/predict";
            } else {
                // Default (English or other)
                url = "https://agri-nav-1.onrender.com/api/predict";
            }


            JSONObject jsonRequest = new JSONObject();
            try {
                jsonRequest.put("query", query);
            } catch (JSONException e) {
                e.printStackTrace();
            }

            JsonObjectRequest request = new JsonObjectRequest(Request.Method.POST, url, jsonRequest,
                    response -> {
                        try {
                            if ("success".equals(response.getString("status"))) {
                                JSONArray predictions = response.getJSONArray("predictions");
                                if (predictions.length() > 0) {
                                    JSONObject firstPrediction = predictions.getJSONObject(0);
                                    String actionMessage = firstPrediction.getString("page_name");
                                    speakResponse(actionMessage);


                                    if (actionMessage.equals("yield_prediction"))
                                    {


                                        String talk;

                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English






                                        if (selectedLanguage.equals("ta")) {
                                            talk = "வணக்கம், உங்கள் பயிர் உற்பத்தி மற்றும் வருமானத்தைப் பற்றி கவலைப்படுகிறீர்கள் என்பதை கேட்டு மகிழ்ச்சி."
                                                    + "அப்படியானால், பயிர் உற்பத்தி கணிப்பு பக்கத்திற்கு செல்லலாம்."
                                                    + "அங்கே நீங்கள் என்ன செய்யவேண்டும் என்று விளக்குகிறேன்: ஆரம்பத்தில் பல இடங்களில் உங்கள் உள்ளீட்டை கேட்கும்."
                                                    + "அங்கு நீங்கள் பயிர் வகை மற்றும் மண் வகை மட்டும் வழங்க வேண்டும், "
                                                    + "மீதமுள்ள உள்ளீடுகள் உங்கள் பண்ணையில் நிறுவிய சென்சார்கள் மூலம் தானாக நிரப்பப்படும்."
                                                    + "விவரங்களைப் பெற்ற பிறகு, சமர்ப்பிக்க பொத்தானை அழுத்தி உங்கள் பயிர் உற்பத்தி கணிப்பு மற்றும் முடிவுகளைப் பார்க்கலாம்."
                                                    + "பயிர் கணிப்பு பக்கத்திற்கு செல்ல, கீழே வலப்பக்க பொத்தானை அழுத்தவும்.";
                                        } else {
                                            talk = "Hello! Glad to know that you care about your crop yield and income. "
                                                    + "Let's move to the Yield Prediction page. "
                                                    + "Here’s what to do: At the start, you'll see multiple fields asking for inputs. "
                                                    + "You only need to provide Crop Type and Soil Type — "
                                                    + "the rest will be automatically filled using your farm’s installed sensors. "
                                                    + "After fetching the details, press the Submit button to view your yield prediction and optimization results. "
                                                    + "To go to the Yield Prediction page, just click the bottom-right button.";
                                        }



                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this,YieldPrediction.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("satellite_analysis"))
                                    {
                                        // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // Tamil message
                                            talk = "வணக்கம்! மேம்பட்ட தொழில்நுட்பங்களைப் பயன்படுத்தி உங்கள் வயலைக் காண விரும்புகிறீர்கள் என்பதை கேட்டு மகிழ்ச்சி. "
                                                    + "உங்கள் உள்ளீடுகளைப் பார்த்தால், எங்களது செயற்கைக்கோள் பகுப்பாய்வு அம்சம் உங்கள் நிலத்தை சிறப்பாக ஆய்வு செய்ய நிச்சயமாக உதவும். "
                                                    + "அதை எப்படி பயன்படுத்துவது என்பதை விளக்குகிறேன்: பக்கத்திற்குள் நுழைந்தவுடன் இரண்டு விருப்பங்கள் காண்பீர்கள். "
                                                    + "முதலில், உங்கள் நிலத்தின் கோஆர்டினேட்டுகள் தெரிந்திருந்தால் அவற்றை நேரடியாக உள்ளிடலாம். "
                                                    + "இல்லையெனில், நீங்கள் உங்கள் நிலத்தில் இருப்பின் 'Current Location' பொத்தானை அழுத்தினால் உங்கள் வயலின் செயற்கைக்கோள் பகுப்பாய்வை காணலாம். "
                                                    + "கவனம்: NDVI மற்றும் NDRE வரைபடங்கள் நைட்ரஜன் அளவை காட்டுகின்றன, மற்றும் Nutrient Map பசளையின்மை உள்ள பகுதிகளை வெளிப்படுத்துகிறது. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Satellite Analysis பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // English message
                                            talk = "Hello! Nice to hear that you want to see your field using advanced techniques. "
                                                    + "From your inputs, I can assure you that our Satellite Analysis feature will definitely help you "
                                                    + "analyze your field in a better way. "
                                                    + "Let me explain what to do: when you enter the page, you will get two options. "
                                                    + "First, if you know your field coordinates, you can directly enter the values; "
                                                    + "otherwise, if you are currently in your field, simply press the 'Current Location' button to see your field’s satellite analysis. "
                                                    + "Note: NDVI and NDRE indicate the nitrogen levels in the crop, and the Nutrient Map shows nutrient-deficient areas. "
                                                    + "Now, press the bottom-right button to go to the Satellite Analysis page.";
                                        }

                                        // Speak in the correct language voice
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, SatelliteAnalysis.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);

                                    }
                                    else if (actionMessage.equals("crop_disease"))
                                    {
                                       // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil message
                                            talk = "வணக்கம்! உங்கள் பயிர்களின் ஆரோக்கியத்தைப் பற்றி நீங்கள் கவலைப்படுகிறீர்கள் என்பதைப் பார்த்து மகிழ்ச்சி. "
                                                    + "இந்த பக்கம், உங்கள் பயிர்களில் மஞ்சள் இலைகள், பழுப்பு புள்ளிகள் அல்லது வளர்ச்சி குறைபாடு போன்ற நோய்களை "
                                                    + "கண்டறிய உதவுவதற்காக வடிவமைக்கப்பட்டுள்ளது. "
                                                    + "பயிர் படத்தை பதிவேற்றிய பிறகு, அமைப்பு நோயைக் கண்டறிந்து அதற்கான சிகிச்சைகள், "
                                                    + "மருந்துகள் மற்றும் தடுப்பு நடவடிக்கைகளை பரிந்துரைக்கும். "
                                                    + "இதனால், எதிர்காலத்திற்காக உங்கள் பயிர்களை பாதுகாக்கலாம். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Crop Disease Detection பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English message
                                            talk = "Hello! I can see that you are concerned about the health of your crops. "
                                                    + "This page is designed to help you detect crop diseases by uploading an image of your crop showing signs such as yellow leaves, brown spots, or stunted growth. "
                                                    + "Once detected, you will receive suggested remedies, treatments, and preventive steps to secure your crops for the future. "
                                                    + "Now, press the bottom-right button to go to the Crop Disease Detection page.";
                                        }

                                        // Speak response in selected language voice
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, CropDisease.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("irrigation_plan"))
                                    {
                                      // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! உங்கள் பண்ணையில் தண்ணீரை புத்திசாலித்தனமாக மேலாண்மை செய்வது மிகவும் முக்கியமானது. "
                                                    + "இந்த பக்கம் அதனைச் செய்ய உங்களுக்கு உதவுகிறது. "
                                                    + "இங்கு, உங்கள் பயிர் வகை, அடுத்த 7 நாட்களுக்கான மழை வாய்ப்பு மற்றும் மண் ஈரப்பதம் ஆகியவற்றின் அடிப்படையில் "
                                                    + "நீர்ப்பாசன அட்டவணைகள் மற்றும் தண்ணீர் விநியோக திட்டங்களைப் பெறுவீர்கள். "
                                                    + "நீங்கள் சில அடிப்படை விவரங்களை மட்டுமே வழங்க வேண்டும்; "
                                                    + "அதன் அடிப்படையில், அமைப்பு உங்களுக்கு சிறந்த நீர்ப்பாசன மதிப்புகள் மற்றும் நேரங்களை பரிந்துரைக்கும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Irrigation Plan பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! It is very important to manage water wisely in your farm, "
                                                    + "and this page will help you do just that. "
                                                    + "Here you will get irrigation schedules and water distribution plans based on your crop type, "
                                                    + "next 7 days’ rainfall chances, and soil moisture data. "
                                                    + "You only need to provide a few basic inputs, "
                                                    + "and the system will suggest optimized watering values and times for your crops. "
                                                    + "Now, press the bottom-right button to go to the Irrigation Plan page.";
                                        }

                                        // Speak response in correct language
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, IrrigationActivity.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("crop_recommendation"))
                                    {
                                        // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! அதிக வருமானத்தைப் பெற எந்த பயிரை வளர்க்கலாம் என்று நீங்கள் யோசித்து வருகிறீர்களா? "
                                                    + "அப்படியானால், இந்தப் பக்கம் உங்களுக்கு மிகவும் பயனுள்ளதாக இருக்கும். "
                                                    + "இந்த அமைப்பு உங்கள் மண் தரவு, வானிலை முறை, மற்றும் நிலைமைகளைப் பயன்படுத்தி "
                                                    + "உங்களுக்கு மிகவும் பொருத்தமான பயிர்களை பரிந்துரைக்கிறது. "
                                                    + "நீங்கள் செய்ய வேண்டியது ஒன்றே — உங்கள் பண்ணையின் தரவுகளை பெற பொத்தானை அழுத்துவது. "
                                                    + "அதன் பிறகு, உங்கள் பண்ணைக்கு தனிப்பயனாக்கப்பட்ட பயிர் பரிந்துரைகள் கிடைக்கும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Crop Recommendation பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! If you are thinking about which crop to plant for maximum benefits, "
                                                    + "then this page will be very useful for you. "
                                                    + "The system uses soil data, weather patterns, and your farm conditions "
                                                    + "to suggest the most suitable crops for your area. "
                                                    + "You just need to click a button to fetch real-time data from your farm, "
                                                    + "and you’ll receive personalized crop recommendations. "
                                                    + "Now, press the bottom-right button to go to the Crop Recommendation page.";
                                        }

                                        // Speak response in the correct language
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, YieldPrediction.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("fertilizer_pesticide_pedia"))
                                    {
                                       // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! சில நேரங்களில் விவசாயிகள் எந்த உரம் அல்லது பூச்சிக்கொல்லியைப் பயன்படுத்த வேண்டும் என்பது பற்றி உறுதியாக இருக்க முடியாது. "
                                                    + "கவலைப்பட வேண்டாம், நாங்கள் உங்களுடன் இருக்கிறோம். "
                                                    + "இந்த பக்கம் அந்த பிரச்சினையைத் தீர்க்க உதவும். "
                                                    + "இங்கே, பல்வேறு உரங்கள் மற்றும் பூச்சிக்கொல்லிகள் பற்றிய விரிவான தகவல்களைப் பெறலாம் — "
                                                    + "அவற்றின் பயன்பாடு, நன்மைகள் மற்றும் பாதுகாப்பு வழிமுறைகள் உட்பட. "
                                                    + "இது உங்களுக்கு சிறந்த முடிவுகளை எடுக்கவும், உங்கள் பயிர்களையும் சுற்றுச்சூழலையும் பாதுகாக்கவும் உதவும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Fertilizer மற்றும் Pesticide Pedia பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! Sometimes farmers are not sure which fertilizer or pesticide to use. "
                                                    + "Don’t worry — we are here to help you. "
                                                    + "This page is designed to solve that problem. "
                                                    + "Here, you’ll find detailed information about various fertilizers and pesticides, "
                                                    + "including their usage, benefits, and safety instructions. "
                                                    + "This will help you make better decisions and protect both your crops and the environment. "
                                                    + "Now, press the bottom-right button to go to the Fertilizer and Pesticide Pedia page.";
                                        }

                                        // Speak response in selected language
                                        speakResponse(talk);
                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, Pedia.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("simulation_game"))
                                    {
                                       // Get saved language
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌱 Tamil version
                                            talk = "வணக்கம்! நீங்கள் எங்கள் சிமுலேஷன் விளையாட்டைப் பார்க்க விரும்புகிறீர்கள் என்று நான் புரிந்துகொண்டேன். "
                                                    + "பயிற்சி மூலம் கற்றல் எப்போதும் சிறந்தது. "
                                                    + "இந்த சிமுலேஷன் விளையாட்டு பக்கம் உங்களுக்கு எந்தவித ஆபத்தும் இல்லாமல் விவசாயத்தைப் பழக வாய்ப்பு தருகிறது. "
                                                    + "நீங்கள் பயிர் வகைகள், பாசனம் மற்றும் உரங்கள் போன்றவற்றில் பல்வேறு முறைகளை முயற்சி செய்து, "
                                                    + "உங்கள் முடிவுகள் உற்பத்தி மற்றும் வருமானத்தில் எப்படி தாக்கம் செலுத்துகின்றன என்பதை காணலாம். "
                                                    + "இது ஒரு வேடிக்கையான மற்றும் பாதுகாப்பான வழி, நவீன வேளாண்மை நுட்பங்களை கற்றுக்கொள்ள. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி Simulation Game பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌱 English version
                                            talk = "Hello! From your input, I can tell you’re interested in exploring our simulation game. "
                                                    + "Learning by doing is always better, and this simulation game lets you practice farming without any real-world risk. "
                                                    + "You can experiment with different crops, irrigation methods, and fertilizers, "
                                                    + "and observe how your choices affect yield and income. "
                                                    + "It’s a fun and safe way to learn modern farming techniques. "
                                                    + "Now, press the bottom-right button to go to the Simulation Game page.";
                                        }

                                        // Speak in the selected language
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, SimulatorActivity.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("farming_guide"))
                                    {
                                       // Get saved language
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! நீங்கள் எங்கள் விவசாய வழிகாட்டியைப் பார்க்க விரும்புகிறீர்கள் என்று தெரிகிறது. "
                                                    + "விவசாய நடைமுறைகளை படிப்படியாக கற்க விரும்பினால், இந்த பக்கம் உங்களுக்கு ஒரு சிறந்த வழிகாட்டி ஆகும். "
                                                    + "இங்கே நெல், கோதுமை, பருத்தி போன்ற பல பயிர்களுக்கான வழிமுறைகள் கிடைக்கும், "
                                                    + "அவை நிலம் தயாரிப்பு, விதைப்பு, பாசனம், பூச்சி கட்டுப்பாடு மற்றும் அறுவடை ஆகியவற்றை உள்ளடக்கியவை. "
                                                    + "இது புதிய மற்றும் அனுபவமுள்ள விவசாயிகளுக்கு தங்கள் முறைகளை மேம்படுத்தவும், உற்பத்தியை அதிகரிக்கவும் உதவும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி விவசாய வழிகாட்டி பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! From your input, I can tell you want to explore our Farming Guide. "
                                                    + "If you wish to learn step-by-step about modern farming practices, this page will be your perfect guide. "
                                                    + "You will find detailed tutorials for crops such as rice, wheat, and cotton, covering soil preparation, sowing, irrigation, pest control, and harvesting. "
                                                    + "This will help both new and experienced farmers improve their methods and productivity. "
                                                    + "Now, press the bottom-right button to go to the Farming Guide page.";
                                        }

                                        // Speak
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, FarmBookPage.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("fertilizer_check"))
                                    {
                                       // Get saved language
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! நீங்கள் எங்கள் உர சரிபார்ப்பு அம்சத்தைப் பயன்படுத்த விரும்புகிறீர்கள் என்று தெரிகிறது. "
                                                    + "போலி அல்லது தரமற்ற உரங்கள் விவசாயிகளுக்கு ஒரு பெரிய பிரச்சனை என்பதை நாங்கள் புரிந்துகொள்கிறோம். "
                                                    + "இந்தப் பக்கம் உங்கள் உரத்தின் உண்மைத்தன்மையை எளிதாக சரிபார்க்க உதவுகிறது. "
                                                    + "உங்கள் உரப் பொதியில் உள்ள பார்கோடு-ஐ ஸ்கேன் செய்யவும், "
                                                    + "அது அரசால் அங்கீகரிக்கப்பட்ட தயாரிப்பா என்பதை முறைமை சரிபார்த்து, "
                                                    + "உங்கள் பயிர்களுக்கு பாதுகாப்பான உரங்களைப் பயன்படுத்துவதை உறுதிசெய்கிறது. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி உர சரிபார்ப்பு பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! I can see that you want to use our Fertilizer Check feature. "
                                                    + "We understand that fake or low-quality fertilizers are a serious problem for farmers. "
                                                    + "This page helps you verify the authenticity of fertilizers easily. "
                                                    + "You just need to scan the barcode on the fertilizer packet, "
                                                    + "and the system will confirm whether it is government-approved or not, ensuring safety for your crops. "
                                                    + "Now press the bottom-right button to go to the Fertilizer Check page.";
                                        }

                                        // Speak
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, FertiCheckActivity.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("community"))
                                    {
                                       // Get saved language
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! நீங்கள் எங்கள் விவசாயிகள் சமூகப் பக்கத்தை அணுக விரும்புகிறீர்கள் என்று தெரிகிறது. "
                                                    + "விவசாயிகள் ஒன்றிணைந்தால் தான் விவசாயம் வலுப்பெறும். "
                                                    + "இந்தப் பக்கம் மூலம் நீங்கள் மற்ற விவசாயிகளுடன் தொடர்புகொள்ளலாம், "
                                                    + "உங்கள் அனுபவங்களைப் பகிரலாம், கேள்விகள் கேட்கலாம், மற்றும் சவால்கள் அல்லது தீர்வுகளை விவாதிக்கலாம். "
                                                    + "இது ஒரு சமூக வலைப்பக்கத்தைப் போன்றது, இதில் அறிவு, ஆலோசனை, மற்றும் ஆதரவு பரிமாறப்படுகிறது. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி சமூக விவாத பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! After listening to you, I can tell you want to access our Community Discussion and Forums. "
                                                    + "Farming becomes stronger when farmers come together, and this page helps you connect with other farmers. "
                                                    + "You can share your experiences, ask questions, and discuss challenges or solutions. "
                                                    + "It’s like a community forum where knowledge, advice, and support are exchanged — so you are never alone in your farming journey. "
                                                    + "Now press the bottom-right button to go to the Community Page.";
                                        }

                                        // Speak the selected language
                                        speakResponse(talk);

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, CommunityMenu.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("government_schemes"))
                                    {
                                        // Get saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! உங்களுக்கு உதவுவதற்கான வாய்ப்பை அளித்ததற்கு நன்றி. "
                                                    + "நீங்கள் அரசு திட்டங்கள் மற்றும் நிதியுதவிகளைப் பற்றி அறிய விரும்புகிறீர்கள் என்று புரிகிறது. "
                                                    + "பல விவசாயிகள் தங்களுக்கு கிடைக்கக்கூடிய அரசு திட்டங்கள் மற்றும் நிதியுதவிகள் குறித்து தெரியாததால் அவற்றை இழக்கிறார்கள். "
                                                    + "இந்தப் பக்கம் மூலம் நீங்கள் அனைத்து அரசு திட்டங்களையும் எளிதாக பார்க்கலாம், "
                                                    + "தகுதி நிபந்தனைகள் மற்றும் விண்ணப்பிக்கும் முறையையும் அறிந்துகொள்ளலாம். "
                                                    + "இதன் மூலம் நீங்கள் பணத்தைச் சேமித்து, நன்மைகளைப் பெற்று, உங்கள் விவசாய நிலையை மேம்படுத்தலாம். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி அரசு திட்டங்கள் பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! Thank you for letting me assist you. "
                                                    + "From what I understand, you want to explore government schemes and subsidies. "
                                                    + "Many farmers miss out on these valuable benefits simply because they are unaware of them. "
                                                    + "This page makes it easy for you to explore all available schemes, check eligibility, and learn how to apply. "
                                                    + "By staying informed, you can save money, gain support, and improve your farming conditions with government help. "
                                                    + "Now press the bottom-right button to go to the Government Schemes page.";
                                        }

                                        // Speak based on language
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, Schemes.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("agri_expert_chatbot"))
                                    {
                                        // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil version
                                            talk = "வணக்கம்! உங்கள் பயிர் உற்பத்தி மற்றும் வருமானத்தைப் பற்றி கவலைப்படுகிறீர்கள் என்பதை கேட்டு மகிழ்ச்சி. "
                                                    + "அப்படியானால், பயிர் உற்பத்தி கணிப்பு பக்கத்திற்கு செல்லலாம். "
                                                    + "அங்கே என்ன செய்ய வேண்டும் என்று விளக்குகிறேன்: ஆரம்பத்தில், சில இடங்களில் உங்கள் உள்ளீட்டை கேட்கப்படும். "
                                                    + "அங்கு நீங்கள் பயிர் வகை மற்றும் மண் வகையை மட்டும் வழங்க வேண்டும். "
                                                    + "மீதமுள்ள விவரங்கள் உங்கள் நிலத்தில் நிறுவப்பட்ட சென்சார்கள் மூலம் தானாக நிரப்பப்படும். "
                                                    + "அனைத்து விவரங்களும் கிடைத்ததும், சமர்ப்பிக்க பொத்தானை அழுத்தி, உங்கள் பயிர் உற்பத்தி கணிப்பு மற்றும் சிறந்த முடிவுகளைப் பார்க்கலாம். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி பயிர் கணிப்பு பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English version
                                            talk = "Hello! I’m glad to hear that you’re concerned about your crop yield and income. "
                                                    + "Let’s go to the Yield Prediction page. "
                                                    + "Here’s what you need to do: at the start, you’ll see several input fields. "
                                                    + "You only need to provide your crop type and soil type. "
                                                    + "The rest of the inputs will be automatically filled using the sensors already installed on your farm. "
                                                    + "After fetching the data, press the submit button to view your yield prediction and optimization results. "
                                                    + "Now press the bottom-right button to go to the Yield Prediction page.";
                                        }

// Speak the selected language
                                        speakResponse(talk);


                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, ChatBot.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("pest_detection"))
                                    {
                                        // Load saved language preference
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en"); // default English

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌿 Tamil version
                                            talk = "வணக்கம்! உங்கள் பயிர்களை பூச்சிகள் தாக்குவதாகக் கேட்டு வருந்துகிறேன். "
                                                    + "பூச்சிகளை விரைவாக கண்டறிவது உங்கள் பயிர்களை காப்பாற்ற மிகவும் முக்கியம். "
                                                    + "இந்தப் பக்கம் அதற்காக உதவுகிறது. "
                                                    + "நீங்கள் பூச்சியின் படத்தை எடுத்தோ அல்லது கேலரியிலிருந்து பதிவேற்றியோ முடியும், "
                                                    + "அதன் பிறகு, அமைப்பு அந்த பூச்சியின் வகையை தானாக அடையாளம் காணும். "
                                                    + "அதோடு, அது எவ்வளவு ஆபத்தானது மற்றும் அதைத் தடுக்கும் பரிந்துரைக்கப்பட்ட முறைகள் பற்றிய தகவல்களையும் வழங்கும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி பூச்சி கண்டறிதல் பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🐛 English version
                                            talk = "Hello! Sorry to hear that you are dealing with pests. "
                                                    + "Identifying pests early can save your crops, and this page will help you do that. "
                                                    + "You can capture or upload a photo of the pest from your gallery, "
                                                    + "and the system will automatically classify the insect. "
                                                    + "It will also tell you how harmful the pest is and the recommended control measures to protect your crops. "
                                                    + "Now press the bottom-right button to go to the Pest Detection page.";
                                        }

                                        // Speak the selected language
                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, PestDetection.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("pest_report"))
                                    {
                                       SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🐜 Tamil Version
                                            talk = "வணக்கம்! பூச்சி தாக்குதலைப் பதிவு செய்ததற்கு நன்றி. "
                                                    + "இந்த நடவடிக்கை மற்ற விவசாயிகளுக்கும் மிகுந்த உதவியாக இருக்கும், "
                                                    + "ஏனெனில் இதன் மூலம் அவர்கள் தங்கள் பயிர்களை பாதுகாக்க தேவையான முன்னெச்சரிக்கை நடவடிக்கைகளை எடுக்க முடியும். "
                                                    + "இந்தப் பக்கம் உங்கள் வயலில் கண்டறியப்பட்ட பூச்சிகளைப் பதிவு செய்ய பயன்படுகிறது. "
                                                    + "இந்த தரவுகள் எதிர்கால பயிர் பாதுகாப்பு திட்டங்களை வடிவமைக்கவும் உதவும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி பூச்சி அறிக்கை பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🪲 English Version
                                            talk = "Hello! Thank you for reporting a pest. "
                                                    + "This action greatly helps other farmers take the right preventive steps. "
                                                    + "This page allows you to report pests detected in your farm. "
                                                    + "The collected reports help others plan better crop protection strategies for the future. "
                                                    + "Now press the bottom-right button to go to the Pest Report page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, PestReportPage.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("pest_map"))
                                    {
                                        SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🐛 Tamil Version
                                            talk = "வணக்கம்! நீங்கள் உங்கள் பகுதியில் உள்ள பூச்சி நிலையை சரிபார்க்க விரும்புகிறீர்கள் என்பதை அறிந்து மகிழ்ச்சி. "
                                                    + "இந்த பூச்சி வரைபடம் உங்கள் நிலத்தில் பூச்சி தாக்குதல்களை காண்பிக்க உதவும். "
                                                    + "வரைபடத்தில் சிவப்பு நிறம் அதிகமாக இருக்கும் பகுதிகள் அதிக பூச்சி தாக்குதல் ஏற்பட்ட பகுதிகளை குறிக்கிறது. "
                                                    + "இதனால் நீங்கள் எவ்விடத்தில் பூச்சி நாசினி பயன்படுத்த வேண்டும் என்பதைக் குறிவைத்து நடவடிக்கை எடுக்கலாம். "
                                                    + "இதனால் தேவையற்ற ரசாயனங்களை வீணாக்காமல் பணமும் சுற்றுச்சூழலும் பாதுகாக்கப்படும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி பூச்சி வரைபடப் பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English Version
                                            talk = "Hello! Nice to hear that you want to check localized pest situations. "
                                                    + "This pest map will help you visualize pest infestations in your field. "
                                                    + "It shows a map with areas where pests are most active — more red means more pest reports from farmers, "
                                                    + "helping you target treatments effectively. "
                                                    + "By using this tool, you can avoid wasting chemicals and only treat affected areas, saving money and protecting the environment. "
                                                    + "Now press the bottom-right button to go to the Pest Map Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, PestMapPage.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("live_market_price"))
                                    {
                                       SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🏷️ Tamil Version
                                            talk = "வணக்கம்! பயிர்களின் தற்போதைய சந்தை விலை பற்றி தெரிந்து கொள்வது மிகவும் முக்கியம். "
                                                    + "இதனால் நீங்கள் உங்கள் விளைபொருட்களை சரியான நேரத்தில் விற்க முடியும். "
                                                    + "இந்த பக்கம் உங்கள் தேர்ந்தெடுத்த பயிர்களின் நேரடி மண்டி விலைகளையும் சந்தை புதுப்பிப்புகளையும் வழங்குகிறது. "
                                                    + "நீங்கள் எளிதாக பல பிராந்தியங்களின் விலைகளை ஒப்பிட்டு, எங்கு விற்கலாம் என்பதைக் கண்டறியலாம். "
                                                    + "இதன் மூலம் அதிக லாபம் ஈட்டவும் இழப்புகளை குறைக்கவும் முடியும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி நேரடி சந்தை விலை பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌾 English Version
                                            talk = "Hello! Knowing the current market price of crops helps you sell at the right time. "
                                                    + "This page provides live mandi rates and market updates for your selected crops. "
                                                    + "You can compare prices from different regions and decide the best place to sell your produce by selecting the crop and region on that page. "
                                                    + "This helps maximize your profit and reduce chances of loss. "
                                                    + "Now press the bottom-right button to go to the Live Market Price Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, MarketPriceActivity.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("agriculture_news"))
                                    {
                                       SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil Version
                                            talk = "அற்புதம்! நீங்கள் அனைத்து வேளாண் செய்திகளையும் இந்த வேளாண் செய்தி பக்கத்தில் காணலாம். "
                                                    + "இது உங்களை விவசாய உலகில் நடைபெறும் சமீபத்திய நிகழ்வுகளுடன் எப்போதும் புதுப்பித்தவனாக வைத்திருக்கும். "
                                                    + "புதிய தொழில்நுட்பங்கள், காலநிலை மாற்றங்கள், அரசாங்க கொள்கைகள் மற்றும் பிற விவசாயிகளின் வெற்றிக் கதைகள் பற்றிய தகவல்களை இங்கே பெறலாம். "
                                                    + "இவ்வாறு புதுப்பிக்கப்பட்டு இருப்பது, சிறந்த முடிவுகளை எடுக்கவும் எதிர்கால சவால்களுக்கு முன்கூட்டியே தயாராகவும் உதவும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி வேளாண் செய்தி பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌱 English Version
                                            talk = "Amazing! You can view all agricultural news on this Agriculture News Page. "
                                                    + "This will keep you updated with the latest happenings in the farming world. "
                                                    + "You will get news about new technologies, weather updates, government policies, and success stories from other farmers. "
                                                    + "Staying updated will help you make better decisions and prepare in advance for any challenges in your farming activities. "
                                                    + "Now press the bottom-right button to go to the Agriculture News Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, NewsPage.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("farms_and_tools"))
                                    {
                                      SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil Version
                                            talk = "வணக்கம்! நீங்கள் உங்கள் விவசாய இயந்திரங்களை நவீன கருவிகள் மற்றும் இயந்திரங்களுடன் மேம்படுத்த விரும்புகிறீர்கள் என்பதை கேட்டு மகிழ்ச்சி. "
                                                    + "இது விவசாயத்தை எளிதாகவும் அதிக உற்பத்தி திறனுடனும் ஆக்கும். "
                                                    + "இந்தப் பக்கம் பல்வேறு விவசாய கருவிகள், உபகரணங்கள் மற்றும் இயந்திரங்கள் பற்றிய தகவல்களை, அவற்றின் பயன்பாடுகள் மற்றும் நன்மைகளுடன் வழங்குகிறது. "
                                                    + "மண்ணை தயாரிப்பதற்கான சிறிய கருவியிலிருந்தும் அறுவடை செய்யும் பெரிய இயந்திரங்களிலிருந்தும், அனைத்தையும் இங்கே கற்றுக்கொண்டு வாங்கவும் முடியும். "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி கருவிகள் மற்றும் இயந்திரங்கள் பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌱 English Version
                                            talk = "Hello! Love to hear that you want to upgrade your machine inventory with modern tools and machinery. "
                                                    + "It can make farming easier and more productive. "
                                                    + "This page provides information about various farming tools, equipment, and machines, along with their uses and benefits. "
                                                    + "Whether it is a small tool for soil preparation or a large machine for harvesting, you can learn about and even purchase them here. "
                                                    + "Now press the bottom-right button to go to the Tools and Machinery Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, NewsPage.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("farmer_to_farmer_trade"))
                                    {
                                       SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil Version
                                            talk = "வணக்கம்! விவசாயிகள் தங்கள் வளங்களைப் பகிர்ந்து, வீணாகாமல் பாதுகாக்க வேண்டும். "
                                                    + "மற்ற விவசாயிகளுடன் நேரடியாக வியாபாரம் செய்வது செலவைக் குறைத்து உங்கள் வருமானத்தை அதிகரிக்க உதவும். "
                                                    + "இந்தப் பக்கம் விவசாயிகளுக்கான சந்தையை உருவாக்குகிறது, இதில் நீங்கள் உங்கள் விளைபொருட்களை விற்கவோ அல்லது மற்றவர்களிடமிருந்து வாங்கவோ முடியும். "
                                                    + "மேலும், விவசாய உபகரணங்கள் மற்றும் வளங்களை பரிமாறவும் முடியும், இதனால் அனைவருக்கும் பயனளிக்கும் சமூக அடிப்படையிலான வணிக அமைப்பு உருவாகிறது. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி விவசாயி-விவசாயி வர்த்தகப் பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌱 English Version
                                            talk = "Hello! Farmers should share their resources to reduce wastage. "
                                                    + "Trading directly with other farmers can save costs and increase your income. "
                                                    + "This page connects you to a farmer-to-farmer marketplace where you can sell your produce or buy crops from others. "
                                                    + "You can also exchange farming inputs and resources, making it a community-based trading system that benefits everyone. "
                                                    + "Now press the bottom-right button to go to the Farmer-to-Farmer Trade Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, YieldPrediction.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }
                                    else if (actionMessage.equals("farmer_dashboard"))
                                    {
                                       SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                        String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                        String talk;

                                        if (selectedLanguage.equals("ta")) {
                                            // 🌾 Tamil Version
                                            talk = "வணக்கம்! உங்கள் கேள்வியை நான் கவனமாகக் கேட்டேன், "
                                                    + "உங்களுக்கு உங்கள் பண்ணையின் புள்ளிவிவரங்களைச் சரிபார்க்க எங்கள் டாஷ்போர்டு தேவை என்பது தெளிவாகத் தெரிகிறது. "
                                                    + "இந்த விவசாயி டாஷ்போர்டு பக்கம் உங்கள் பண்ணையின் முழுமையான பார்வையை வழங்குகிறது. "
                                                    + "இங்கே உங்கள் சுயவிவரம், புள்ளிவிவரங்கள், விளைச்சல் தரவு மற்றும் மொத்த பண்ணை சுருக்கம் அனைத்தையும் ஒரே இடத்தில் பார்க்க முடியும். "
                                                    + "இது உங்கள் தனிப்பட்ட கட்டுப்பாட்டு அறை போல, முக்கியமான தகவல்களை ஒரே இடத்தில் சேகரித்து, "
                                                    + "உங்கள் வேளாண் செயல்பாடுகளை சிறப்பாக திட்டமிடவும் ஒழுங்குபடுத்தவும் உதவுகிறது. "
                                                    + "இப்போது கீழே வலப்பக்க பொத்தானை அழுத்தி டாஷ்போர்டு பக்கத்திற்கு செல்லுங்கள்.";
                                        } else {
                                            // 🌱 English Version
                                            talk = "Hello! I carefully listened to your query, and I can easily tell that you need our dashboard to check all your farm stats. "
                                                    + "This farmer dashboard page gives you a complete overview of your farm. "
                                                    + "You can view your profile, statistics, yield data, and overall farm summary in one place. "
                                                    + "It acts like your personal control room, where all important information is gathered to help you stay organized "
                                                    + "and plan your farming activities more efficiently. "
                                                    + "Now press the bottom-right button to go to the Dashboard Page.";
                                        }

                                        speakResponse(talk);

                                        go.setOnClickListener(new View.OnClickListener() {
                                            @Override
                                            public void onClick(View v) {
                                                startActivity(new Intent(AgriMam.this, Dashboard.class));
                                            }
                                        });

                                        go.setVisibility(View.VISIBLE);
                                    }




                                } else {
                                   SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                    String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                    String talk;

                                    if (selectedLanguage.equals("ta")) {
                                        // 🌾 Tamil Version
                                        talk = "மன்னிக்கவும், உங்கள் கேள்வியை நான் புரிந்துகொள்ள முடியவில்லை. தயவுசெய்து மீண்டும் முயற்சி செய்யவும் அல்லது தெளிவாகச் சொல்லுங்கள்.";
                                    } else {
                                        // 🌱 English Version
                                        talk = "Sorry, I could not understand your query. Please try again or say it more clearly.";
                                    }

                                    speakResponse(talk);

                                    go.setOnClickListener(new View.OnClickListener() {
                                        @Override
                                        public void onClick(View v) {
                                            startActivity(new Intent(AgriMam.this, YieldPrediction.class));
                                        }
                                    });

                                    go.setVisibility(View.VISIBLE);

                                }
                            } else {
                               SharedPreferences prefs = getSharedPreferences("AppPrefs", MODE_PRIVATE);
                                String selectedLanguage = prefs.getString("SelectedLanguage", "en");

                                String talk;

                                if (selectedLanguage.equals("ta")) {
                                    // 🌾 Tamil Version
                                    talk = "மன்னிக்கவும், சர்வரில் இருந்து பதில் பெற முடியவில்லை. உங்கள் இணைய இணைப்பைச் சரிபார்த்து மீண்டும் முயற்சிக்கவும்.";
                                } else {
                                    // 🌱 English Version
                                    talk = "Sorry, failed to get a response from the server. Please check your internet connection and try again.";
                                }

                                speakResponse(talk);

                                go.setOnClickListener(new View.OnClickListener() {
                                    @Override
                                    public void onClick(View v) {
                                        startActivity(new Intent(AgriMam.this, YieldPrediction.class));
                                    }
                                });

                                go.setVisibility(View.VISIBLE);


                            }
                        } catch (JSONException e) {
                            e.printStackTrace();
                            speakResponse("Error parsing server response.");
                        }
                    },
                    error -> {
                        error.printStackTrace();
                        speakResponse("Failed to connect to server.");
                    });

            requestQueue.add(request);
        }

        private void speakResponse(String text) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "tts_response");
        }

        @Override
        protected void onDestroy() {
            super.onDestroy();
            if (tts != null) {
                tts.stop();
                tts.shutdown();
            }
        }


    }