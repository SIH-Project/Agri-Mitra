package com.example.agrimitra;



import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

public class IrrigationActivity extends AppCompatActivity {

    private static final String TAG = "IrrigationActivity";
    private TranslationHelper translationHelper;

    TextView heading,text1,tvFarmPlan;
    
    ProgressBar progressBar;
    private static final String API_KEY_TOMORROW = "w9CRWiUd2tvXF8MQO12jYQVnJKX6CwFq";
    private static final String API_URL_OPENROUTER = "https://openrouter.ai/api/v1/chat/completions";
    //    private static final String API_KEY_OPENROUTER = "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";
//
    private static final String API_KEY_OPENROUTER = "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";

    private RecyclerView recyclerView;
    private PrecipitationAdapter adapter;
    private TextView result_textview;

    private OkHttpClient client;

    // Weather data placeholders
    private String d1 = "", d2 = "", d3 = "", d4 = "", d5 = "", d6 = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_irrigation);

        TextView heading = findViewById(R.id.heading);
        TextView text1 = findViewById(R.id.text1);
        TextView tvFarmPlan = findViewById(R.id.tvFarmPlan);




        // Initialize translation helper
        translationHelper = new TranslationHelper(this);


        translationHelper.initializeTranslation(() -> {
            translationHelper.translateTextView(heading, "Weekly Precipitation Plan");
            translationHelper.translateTextView(text1, "Irrigation Plan");
            translationHelper.translateTextView(tvFarmPlan, "Please wait...");
        });



        progressBar=findViewById(R.id.progress);

        result_textview = findViewById(R.id.tvFarmPlan);
        recyclerView = findViewById(R.id.recyclerViewPrecipitation);
        recyclerView.setLayoutManager(new LinearLayoutManager(this));

        // Setup OkHttpClient with logging
        HttpLoggingInterceptor logging = new HttpLoggingInterceptor(message -> Log.d("HTTP_LOGS", message));
        logging.setLevel(HttpLoggingInterceptor.Level.BODY);
        client = new OkHttpClient.Builder().addInterceptor(logging).build();

        // Start fetching weather data
        fetchWeatherData();
    }

    private void fetchWeatherData() {
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl("https://api.tomorrow.io/")
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        TomorrowApiService apiService = retrofit.create(TomorrowApiService.class);

        Call<TomorrowResponse> call = apiService.getDailyForecast(
                "13.0827,80.2707", // Chennai coordinates
                "precipitationIntensity,precipitationProbability",
                "1d",
                "metric",
                API_KEY_TOMORROW
        );

        call.enqueue(new Callback<TomorrowResponse>() {
            @Override
            public void onResponse(Call<TomorrowResponse> call, Response<TomorrowResponse> response) {
                if (response.isSuccessful() && response.body() != null) {
                    List<TomorrowResponse.Interval> intervals =
                            response.body().data.timelines.get(0).intervals;

                    // Set adapter
                    adapter = new PrecipitationAdapter(IrrigationActivity.this,intervals);
                    recyclerView.setAdapter(adapter);

                    // Extract precipitation probability
                    d1 = String.valueOf(intervals.get(0).values.precipitationProbability);
                    d2 = String.valueOf(intervals.get(1).values.precipitationProbability);
                    d3 = String.valueOf(intervals.get(2).values.precipitationProbability);
                    d4 = String.valueOf(intervals.get(3).values.precipitationProbability);
                    d5 = String.valueOf(intervals.get(4).values.precipitationProbability);
                    d6 = String.valueOf(intervals.get(5).values.precipitationProbability);

                    // ✅ Call OpenRouter only after weather data is ready
                    sendPromptToOpenRouter();

                } else {
                    Log.e(TAG, "Tomorrow.io API Response failed: " + response.code());
                }
            }

            @Override
            public void onFailure(Call<TomorrowResponse> call, Throwable t) {
                Log.e(TAG, "Tomorrow.io API Call failed", t);
            }
        });
    }

    private void sendPromptToOpenRouter() {
        MediaType JSON = MediaType.parse("application/json; charset=utf-8");

        JSONObject json = new JSONObject();
        try {
            json.put("model", "openai/gpt-oss-20b:free");

            JSONArray messages = new JSONArray();
            JSONObject userMsg = new JSONObject();
            userMsg.put("role", "user");
            userMsg.put("content",
                    "day1:" + d1 +
                            " day2:" + d2 +
                            " day3:" + d3 +
                            " day4:" + d4 +
                            " day5:" + d5 +
                            " day6:" + d6 +
                            " based on these precipitation chances of next 6 days, give me irrigation plan for potato farm, start the result with Hello Farmer,Note: Do not use tables and any diagrams, just give result as text, do not use ** ,give me day1,day2,day3,day4,day5,day6 irrigation plan per acre summary");
            messages.put(userMsg);

            json.put("messages", messages);

        } catch (JSONException e) {
            e.printStackTrace();
        }

        RequestBody body = RequestBody.create(json.toString(), JSON);

        Request request = new Request.Builder()
                .url(API_URL_OPENROUTER)
                .post(body)
                .addHeader("Authorization", "Bearer " + API_KEY_OPENROUTER)
                .addHeader("Content-Type", "application/json")
                .build();

        client.newCall(request).enqueue(new okhttp3.Callback() {
            @Override
            public void onFailure(okhttp3.Call call, IOException e) {



                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    result_textview.setVisibility(View.VISIBLE);
                    translationHelper.translateText("Error: " + e.getMessage(), translatedText -> {
                        result_textview.setText(translatedText);
                    });
                });
                Log.e("OpenRouter_Error", e.getMessage(), e);
            }

            @Override
            public void onResponse(okhttp3.Call call, okhttp3.Response response) throws IOException {
                if (!response.isSuccessful()) {
                    runOnUiThread(() -> {
                        translationHelper.translateText("Error: " + response.message(), translatedText -> {
                            result_textview.setText(translatedText);
                        });
                    });
                    return;
                }





                String responseData = response.body().string();
                Log.d("OpenRouter_Response", responseData);

                try {
                    JSONObject jsonResponse = new JSONObject(responseData);
                    JSONArray choices = jsonResponse.getJSONArray("choices");
                    JSONObject message = choices.getJSONObject(0).getJSONObject("message");
                    String result = message.getString("content");

                    // Trim everything before "Hello Farmer,"
                    int startIndex = result.lastIndexOf("Hello Farmer,");
                    if (startIndex != -1) result = result.substring(startIndex);

                    final String finalResult = result.trim();

                    runOnUiThread(() -> {
                        progressBar.setVisibility(View.GONE);
                        result_textview.setVisibility(View.VISIBLE);
                        result_textview.setText(finalResult);
                    });



                } catch (JSONException e) {
                    runOnUiThread(() -> {
                        translationHelper.translateText("Parse error: " + responseData, translatedText -> {
                            result_textview.setText(translatedText);
                        });
                    });
                }
            }
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
